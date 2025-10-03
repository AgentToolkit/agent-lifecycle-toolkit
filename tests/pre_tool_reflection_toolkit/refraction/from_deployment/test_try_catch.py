from nestful.data_handlers import get_nestful_catalog
from altk.pre_tool_reflection_toolkit.refraction.src import refract


class TestTryCatch:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)
        self.tokens = ["made up tokens"]
        self.memory = ["error inducing memory"]

    def test_basic(self) -> None:
        result = refract(
            sequence=self.tokens, catalog=self.catalog, memory_objects=self.memory
        )  # type: ignore
        assert result.error

    def test_with_timeout(self) -> None:
        result = refract(
            sequence=self.tokens,
            catalog=self.catalog,
            memory_objects=self.memory,
            timeout=1.0,
        )  # type: ignore
        assert result.error
