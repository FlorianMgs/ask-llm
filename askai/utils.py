import re

from contextlib import suppress
from inspect import getfullargspec
from typing import Any, Callable, Iterable

from jinja2 import Environment, BaseLoader
from langchain.output_parsers import PydanticOutputParser

from .schemas import BaseAnswer


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


def format_return_type_instructions(schema: Any, raw_template: str) -> str:
    """
    Returns the format instructions of a Pydantic schema.
    """
    if schema and issubclass(schema, BaseAnswer):
        raw_template += "\n\n" + output_parser(schema).get_format_instructions()
    return raw_template


def get_nested_attr(obj: Any, attr: str) -> Any:
    """
    Recusrively gets a nested attribute from an object.
    """
    for attr in attr.split("."):
        if hasattr(obj, attr) and not callable(getattr(obj, attr)):
            obj = getattr(obj, attr)
    return obj


def output_parser(schema: BaseAnswer) -> PydanticOutputParser:
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


def render_template(context_dict: dict, raw_template: str) -> str:
    """
    Render the raw template with Jinja
    """
    environment = Environment(loader=BaseLoader())
    template = environment.from_string(raw_template)
    return template.render(**context_dict)
