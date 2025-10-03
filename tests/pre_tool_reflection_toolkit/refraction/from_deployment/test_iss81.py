from nestful.data_handlers import get_nestful_catalog
from nestful import SequenceStep
from altk.pre_tool_reflection_toolkit.refraction.src import refract


class TestISS81:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    def test_empty_suggestion(self) -> None:
        call = {
            "name": "Real-Time_Product_Search_Product_Reviews",
            "arguments": {"limit": 10, "country": "us", "language": "en"},
            "label": "var2",
        }

        memory = {
            "var1": {
                "product_id": "INIT",
                "product_title": "INIT",
                "product_description": "INIT",
                "product_photos": "INIT",
                "product_attributes": "INIT",
                "product_rating": "INIT",
                "product_page_url": "INIT",
                "product_offers_page_url": "INIT",
                "product_specs_page_url": "INIT",
                "product_reviews_page_url": "INIT",
                "product_num_reviews": "INIT",
                "product_num_offers": "INIT",
            }
        }

        step = SequenceStep(**call)

        result = refract(
            sequence=step,
            catalog=self.catalog,
            memory_objects=memory,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
