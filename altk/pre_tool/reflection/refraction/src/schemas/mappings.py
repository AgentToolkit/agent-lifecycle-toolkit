from __future__ import annotations
from nl2flow.compile.schemas import MappingItem
from pydantic import BaseModel
from typing import Optional


class Mapping(MappingItem):
    pass


class MappingCandidate(BaseModel):
    name: str
    description: str
    type: Optional[str] = None
    source: str
    is_input: bool


class MappingLabel(BaseModel):
    label: str
    map: Optional[str] = None
