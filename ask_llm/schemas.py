import json
from typing import Any, Generator
from langchain_core.pydantic_v1 import BaseModel


class BaseAnswer(BaseModel):
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)

    def __iter__(self) -> Generator:
        for field in self.__fields__:
            yield field, getattr(self, field)

    def __len__(self) -> int:
        return len(self.__fields__)

    def to_dict(self) -> dict:
        return json.loads(self.json())
