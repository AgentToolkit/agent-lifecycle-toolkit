from nestful.data_handlers import get_nestful_catalog
from altk.pre_tool_reflection_toolkit.refraction.src.batch_actions import run_all_modes
from tests.utils.refraction.utils import get_cached_maps

import pytest

pytestmark = pytest.mark.refract_extra


class TestMultipleFixes:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog()
        self.cached_mappings = get_cached_maps()

    def test_multiple_fixes(self) -> None:
        run_all_modes(
            sequence=[
                {
                    "name": "Spotify_Scraper_Get_Artist_Overview",
                    "arguments": {"artistId": "$var1.id$"},
                }
            ],
            catalog=self.catalog,
            mappings=self.cached_mappings,
            memory_objects={
                "var1": {
                    "artist_id": 12345,
                }
            },
        )
