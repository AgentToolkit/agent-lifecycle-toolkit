# from nestful.data_handlers import get_nestful_catalog
# from refraction import refract
# from tests.nestful_experiments.utils import get_cached_maps
#
#
# class TestCC:
#     def setup_method(self) -> None:
#         self.catalog = get_nestful_catalog(executable=True)
#         self.mappings = get_cached_maps()
#
#     def test_cc_filter(self) -> None:
#         pass
#
#     def test_timeout(self) -> None:
#         pass
#
#     def test_pass(self) -> None:
#         memory = {
#             "var1": {},
#         }
#
#         sequence = [
#             'var5 = TripadvisorSearchHotels(checkIn="2024-08-15",'
#             ' checkOut="2024-08-18")'
#         ]
#
#         result = refract(
#             sequence,
#             catalog=self.catalog,
#             mappings=self.mappings,
#             memory_objects=memory,
#             use_given_operators_only=True,
#         )
#
#         assert result.report.determination is False
#         assert result.is_timeout is False
#
