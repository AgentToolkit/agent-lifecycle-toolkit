from typing_extensions import TypedDict
from altk.pre_tool_reflection_toolkit.refraction.src.integration import (
    refract,
    Refractor,
)
from typing import Dict, Any, Optional


class State(TypedDict):
    query: str
    geoId: str
    checkIn: str
    checkOut: str
    response: Dict[str, Any]
    refractor: Optional[Refractor]
    memory: Dict[str, Any]


@refract(
    api="TripadvisorSearchHotels",
    use_given_operators_only=False,
    execute_if_fixed=True,
    use_state=True,
)
def search_hotels(state: State) -> Dict[str, Any]:
    """Tripadvisor search hotels"""
    return {"response": f"{state.get('geoId')}: hotels", **state}


@refract()
def TripadvisorSearchLocation(query: str) -> str:
    """Tripadvisor search location"""
    return f"{query}: locations"
