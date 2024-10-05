import os
import json

from copy import deepcopy
from functools import wraps
from inspect import signature, isclass
from pprint import pprint
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from .images import convert_image_to_base64
from pydantic import BaseModel
from .utils import (
    clean_docstring,
    format_args,
    format_attributes,
    format_func_body_result,
    format_return_type_instructions,
    output_parser,
    render_chat_messages,
)


load_dotenv()


def ask(
    call: bool = True,
    verbose: bool = False,
    return_prompt_only: bool = False,
    return_as_dict: bool = False,
    model_name: str = "gpt-4o",
    max_tokens: int = 4096,
    image_quality: str = "high",
    chat_model_class: BaseChatModel = ChatOpenAI,
    **llm_kwargs,
):
    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
            api_key_from_kwargs = kwargs.pop("api_key", None)
            context_dict = deepcopy(kwargs)
            image = context_dict.get("image", None)
            return_type = signature(func).return_annotation
            docstring = func.__doc__
            messages = []

            is_claude = model_name.startswith("claude")
            is_o1 = model_name.startswith("o1")
            bypass_system_msg = is_claude or is_o1

            if is_claude:
                chat_model_class = ChatAnthropic
                api_key_from_llm_kwargs = llm_kwargs.get("anthropic_api_key", None)
                llm_kwargs["anthropic_api_key"] = (
                    api_key_from_kwargs
                    or api_key_from_llm_kwargs
                    or os.getenv("ANTHROPIC_API_KEY")
                )

            elif model_name.startswith("gpt") or is_o1:
                chat_model_class = ChatOpenAI
                api_key_from_llm_kwargs = llm_kwargs.get("openai_api_key", None)
                llm_kwargs["openai_api_key"] = (
                    api_key_from_kwargs
                    or api_key_from_llm_kwargs
                    or os.getenv("OPENAI_API_KEY")
                )

            if verbose:
                print("RETURN TYPE: ", return_type)

            if docstring:
                raw_template = clean_docstring(docstring)
                raw_template, context_dict = format_func_body_result(
                    result=func(*args, **kwargs),
                    raw_template=raw_template,
                    context_dict=context_dict,
                )

                raw_template = format_return_type_instructions(
                    return_type=return_type, raw_template=raw_template
                )

                raw_template, context_dict = format_attributes(
                    func=func,
                    func_args=args,
                    context_dict=context_dict,
                    raw_template=raw_template,
                )

                context_dict = format_args(
                    func=func, func_args=args, context_dict=context_dict
                )
                messages.extend(
                    render_chat_messages(
                        context_dict,
                        raw_template,
                        bypass_system=bypass_system_msg,
                    )
                )

            else:
                msg_class = SystemMessage if not bypass_system_msg else HumanMessage
                messages.append(
                    msg_class(content=[{"type": "text", "text": " ".join(args)}])
                )

            if image:
                image_base64 = convert_image_to_base64(image)
                messages.append(
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64,
                                    "detail": image_quality,
                                },
                            }
                        ]
                    )
                )

            if verbose:
                print("FORMATTED PROMPT MESSAGES:")
                for message in messages:
                    pprint(message.dict())

            prompt = ChatPromptTemplate.from_messages(messages)
            if return_prompt_only:
                return prompt

            llm_model_class_kwargs = {
                "model_name": model_name,
                "max_tokens": max_tokens,
                **llm_kwargs,
            }
            if is_o1:
                llm_model_class_kwargs.pop("max_tokens")
                llm_model_class_kwargs["temperature"] = 1

            llm = chat_model_class(**llm_model_class_kwargs)
            if (
                return_type
                and isclass(return_type)
                and issubclass(return_type, BaseModel)
            ):
                parser = output_parser(return_type)
                retry_parser = RetryWithErrorOutputParser.from_llm(
                    llm=llm, parser=parser
                )
                completion_chain = prompt | llm
                main_chain = (
                    RunnableParallel(
                        completion=completion_chain,
                        prompt_value=prompt,
                    )
                    | RunnableLambda(
                        lambda x: retry_parser.parse_with_prompt(
                            completion=x.get("completion").content,
                            prompt_value=x.get("prompt_value"),
                        )
                    )
                    | RunnableLambda(
                        lambda x: json.loads(x.json()) if return_as_dict else x
                    )
                )
                return main_chain.invoke({}) if call else main_chain

            elif (
                return_type
                and not isclass(return_type)
                and return_type != str
                and return_type != Any
                and return_type != None
            ):
                return eval((prompt | llm).invoke({}).content) if call else prompt | llm

            else:
                return (prompt | llm).invoke({}).content if call else prompt | llm

        return wrapper

    return decorator
