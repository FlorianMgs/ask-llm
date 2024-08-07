import os
import json

from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Callable

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

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

            if docstring:
                raw_template = clean_docstring(docstring)
                raw_template, context_dict = format_func_body_result(
                    result=func(*args, **kwargs),
                    raw_template=raw_template,
                    context_dict=context_dict,
                )

                raw_template = format_return_type_instructions(
                    schema=return_type, raw_template=raw_template
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
                messages.extend(render_chat_messages(context_dict, raw_template))

            else:
                messages.append(SystemMessage(content=[{"type": "text", "text": " ".join(args)}]))

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

            prompt = ChatPromptTemplate.from_messages(messages)
            if return_prompt_only:
                return prompt

            if issubclass(chat_model_class, ChatOpenAI):
                api_key_from_llm_kwargs = llm_kwargs.get("openai_api_key", None)
                llm_kwargs["openai_api_key"] = (
                    api_key_from_kwargs
                    or api_key_from_llm_kwargs
                    or os.getenv("OPENAI_API_KEY")
                )

            llm = chat_model_class(
                model_name=model_name,
                max_tokens=max_tokens,
                **llm_kwargs,
            )
            if return_type and issubclass(return_type, BaseModel):
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
                    | RunnableLambda(lambda x: json.loads(x.json()) if return_as_dict else x)
                )
                return main_chain.invoke({}) if call else main_chain

            return (prompt | llm).invoke({}).content if call else prompt | llm

        return wrapper

    return decorator
