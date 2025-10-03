from altk.pre_tool_reflection_toolkit.refraction.src.integration import refract
from typing import Dict, Any, Optional


@refract()
def SkyScrapperFlightSearch(
    originSkyId: str,
    destinationSkyId: str,
    originEntityId: str,
    destinationEntityId: str,
    date: str,
    returnDate: Optional[str] = None,
    cabinClass: Optional[str] = "economy",
    adults: Optional[int] = 1,
    children: Optional[int] = 0,
    infants: Optional[int] = 0,
    sortBy: Optional[str] = "best",
    limit: Optional[int] = None,
    carriersIds: Optional[str] = None,
    currency: Optional[str] = "USD",
    market: Optional[str] = "en-US",
    countryCode: Optional[str] = "US",
) -> Dict[str, Any]:
    """SkyScrapper flight search"""
    return {"flightId": 12345}


@refract()
def TripadvisorSearchLocation(
    query: str,
) -> Dict[str, Any]:
    return {"query": query, "geoId": 123}


@refract(
    api="TripadvisorSearchHotels",
    use_given_operators_only=False,
    execute_if_fixed=True,
)
def search_hotels(**kwargs: Any) -> Dict[str, Any]:
    return {"id": "hotel123", **kwargs}
