from nl2flow.compile.schemas import MemoryItem
from typing import Dict, Any


class MemoryObject(MemoryItem):
    value: Dict[str, Any]
