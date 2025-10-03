from nestful import Catalog, API
from nl2flow.compile.flow import Flow
from typing import Set, Dict
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components


def reduce_to_cc(
    flow: Flow,
    catalog: Catalog,
) -> Catalog:
    all_variables: Set[str] = set()

    for api in catalog.apis:
        new_variables = variables_involved_in_api(api)
        all_variables = {*all_variables, *new_variables}

    variable_map: Dict[str, Set[str]] = {item: {item} for item in all_variables}

    for item in variable_map:
        maps_to = list(
            filter(
                lambda x: x.source_name == item,
                flow.flow_definition.list_of_mappings,
            )
        )

        for mapping in maps_to:
            variable_map[item].add(mapping.target_name)

    relation_matrix = []

    for reference_api in catalog.apis:
        relation_vector = []

        for target_api in catalog.apis:
            if reference_api.name == target_api.name:
                relation_vector.append(0)
            else:
                reference_variables = reference_api.get_outputs()

                for item in reference_variables:
                    reference_variables = {
                        *reference_variables,
                        *variable_map[item],
                    }

                target_variables = target_api.get_arguments(required=True)

                for item in target_variables:
                    target_variables = {*target_variables, *variable_map[item]}

                if any([item in target_variables for item in reference_variables]):
                    relation_vector.append(1)

                else:
                    relation_vector.append(0)

        relation_matrix.append(relation_vector)

    graph = csr_array(relation_matrix)
    _, labels = connected_components(csgraph=graph, directed=True, return_labels=True)

    # _ = {
    #     int(label): {
    #         catalog.apis[index].name
    #         for index, ll in enumerate(labels)
    #         if ll == label
    #     }
    #     for label in labels
    # }

    new_catalog = Catalog()
    return new_catalog


def variables_involved_in_api(api: API) -> Set[str]:
    variables = {*api.get_arguments(), *api.get_outputs()}

    for item in api.output_parameters:
        properties = api.output_parameters[item].properties
        variables = {*variables, *properties.keys()}

    return variables
