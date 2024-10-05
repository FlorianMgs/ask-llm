import re

from contextlib import suppress
from inspect import getfullargspec, isclass
from textwrap import dedent
from typing import Any, Callable, Iterable

from jinja2 import Environment, BaseLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from pydantic import BaseModel


def preprocess_template(template_str):
    # Remove custom tags {% chat %} and {% endchat %}
    template_str = re.sub(r"{%\s*chat\s*%}", "", template_str)
    template_str = re.sub(r"{%\s*endchat\s*%}", "", template_str)
    # Replace {% message 'role' %} and {% endmessage %} with placeholders
    template_str = re.sub(
        r"{%\s*message\s*(\w+)\s*%}", r"<!-- message \1 -->", template_str
    )
    template_str = re.sub(r"{%\s*endmessage\s*%}", r"<!-- endmessage -->", template_str)
    return template_str


def render_chat_messages(
    context_dict: dict,
    raw_template: str,
    bypass_system: bool = False,  # Claude / o1 does not support SystemMessage
) -> list:
    environment = Environment(loader=BaseLoader())
    preprocessed_template = preprocess_template(raw_template)
    template = environment.from_string(preprocessed_template)
    rendered_template = dedent(template.render(**context_dict))

    messages = []

    # Parse the text outside chat tags and add it as a HumanMessage
    outside_chat_pattern = re.compile(r"(.*?)<!-- message", re.DOTALL)
    outside_chat_match = outside_chat_pattern.match(rendered_template)
    if outside_chat_match:
        outside_text = outside_chat_match.group(1).strip()
        if outside_text:
            messages.append(
                HumanMessage(content=[{"type": "text", "text": outside_text}])
            )

    # Parse the rendered template to construct the list of message objects
    message_pattern = re.compile(
        r"<!-- message (\w+) -->\s*(.*?)\s*<!-- endmessage -->", re.DOTALL
    )
    role_map = {
        "system": "SystemMessage" if not bypass_system else "HumanMessage",
        "ai": "AIMessage",
        "human": "HumanMessage",
    }

    for match in message_pattern.finditer(rendered_template):
        role, content = match.groups()
        message_class = role_map.get(role, "UnknownMessage")
        content = content.strip()
        if message_class == "SystemMessage":
            messages.append(SystemMessage(content=[{"type": "text", "text": content}]))
        elif message_class == "AIMessage":
            messages.append(AIMessage(content=[{"type": "text", "text": content}]))
        elif message_class == "HumanMessage":
            messages.append(HumanMessage(content=[{"type": "text", "text": content}]))
        else:
            raise ValueError(f"Unknown role: {role}")

    # Add remaining text outside chat tags as a HumanMessage
    remaining_text = message_pattern.split(rendered_template)[-1].strip()
    if remaining_text:
        messages.append(
            HumanMessage(content=[{"type": "text", "text": remaining_text}])
        )

    return messages


def format_attributes(
    func: Callable, func_args: Iterable, context_dict: dict, raw_template: str
) -> tuple[str, dict]:
    """
    Formats the attributes of the method's parent class / obj.
    """
    varnames = func.__code__.co_varnames
    if "self" in varnames or "cls" in varnames:
        _reference = "self" if "self" in varnames else "cls"
        _object = func_args[0]
        placeholders = re.findall(r"\{\{(.*?)\}\}|\{%\s*(.*?)\s*%\}", raw_template)
        placeholders = [
            item.strip()
            for sublist in placeholders
            for item in sublist
            if item and _reference in item
        ]
        placeholders = list(set(placeholders))
        for placeholder in placeholders:
            _attribute = placeholder.split(f"{_reference}.")[-1]
            if "." in _attribute:
                attr_template_name = _attribute.replace(".", "__")
            else:
                attr_template_name = _attribute
            context_dict[attr_template_name] = get_nested_attr(_object, _attribute)
            raw_template = raw_template.replace(
                f"{_reference}.{_attribute}", attr_template_name
            )
    return raw_template, context_dict


def format_args(func: Callable, func_args: Iterable, context_dict: dict) -> dict:
    """
    Formats the arguments of a function.
    """
    if len(func_args) > 0:
        args_names = getfullargspec(func).args
        self_is_here = "self" in args_names
        if self_is_here:
            args_names = args_names[1:]
        for i, arg_name in enumerate(args_names):
            with suppress(IndexError):
                context_dict[arg_name] = func_args[i + 1 if self_is_here else i]
    return context_dict


def format_func_body_result(
    result: Any, raw_template: str, context_dict: dict
) -> tuple[str, dict]:
    """
    Formats the result of a function to be used in a prompt.
    """
    if result and "__result__" in raw_template:
        context_dict["result_"] = result
        raw_template = raw_template.replace("__result__", "result_")
    return raw_template, context_dict


def format_return_type_instructions(return_type: Any, raw_template: str) -> str:
    """
    Returns the format instructions of a Pydantic schema.
    """
    if return_type and isclass(return_type) and issubclass(return_type, BaseModel):
        raw_template += "\n\n" + output_parser(return_type).get_format_instructions()
    elif (
        return_type
        and not isclass(return_type)
        and return_type != Any
        and return_type != None
        and return_type != str
    ):
        raw_template += (
            "\n\n"
            "Return the result as a string representing a perfectly exact python expression "
            f"respecting this type annotation: {return_type}\n"
            "Answer ONLY by the python expression, without any additional text. "
            "Do not enclose your answer in markdown code blocks."
        )
    return raw_template


def get_nested_attr(obj: Any, attr: str) -> Any:
    """
    Recusrively gets a nested attribute from an object.
    """
    for attr in attr.split("."):
        if hasattr(obj, attr) and not callable(getattr(obj, attr)):
            obj = getattr(obj, attr)
    return obj


def output_parser(schema: BaseModel) -> PydanticOutputParser:
    """
    Converts a Pydantic scheme to a LCEL compatible output parser.
    Can be used to either get format instructions in a prompt or to parse the llm output.
    """
    return PydanticOutputParser(pydantic_object=schema)


def clean_docstring(docstring: str) -> str:
    """
    Cleans a docstring from unnecessary spaces and new lines.
    """
    docstring = docstring.strip()
    docstring = re.sub(" +", " ", docstring)
    docstring = re.sub("\n+", "\n", docstring)
    return docstring
