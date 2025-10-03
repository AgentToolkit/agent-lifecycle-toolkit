from nl2flow.printers.driver import Printer
from nl2flow.plan.schemas import ClassicalPlan as Plan, Action
from nl2flow.compile.schemas import Step, ClassicalPlanReference
from nl2flow.compile.options import BasicOperations
from nl2flow.compile.basic_compilations.compile_references.utils import (
    get_token_predicate_name,
)

from nestful import SequencingData, API, Catalog
from nestful.utils import extract_label, parse_parameters
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.mappings import (
    MappingLabel,
)
from typing import Union, Any, Dict, List
from warnings import warn
from re import match


class CustomPrint(Printer):
    @classmethod
    def add_to_current_map(
        cls,
        current_label: str,
        memory_item: Dict[str, Any],
        current_maps: Dict[str, str],
    ) -> None:
        for key, value in memory_item.items():
            if isinstance(value, Dict):
                current_label = f"{current_label}.{key}"
                cls.add_to_current_map(current_label, value, current_maps)
            else:
                current_maps[key] = f'"${current_label}.{key}$"'

    @classmethod
    def set_memory(cls, memory: Dict[str, Any]) -> Dict[str, str]:
        current_maps: Dict[str, str] = {}

        for label, memory_item in memory.items():
            if label.startswith("var"):
                if isinstance(memory_item, Dict):
                    cls.add_to_current_map(label, memory_item, current_maps)
                else:
                    current_maps[label] = label

            else:
                pass

        return current_maps

    @staticmethod
    def add_extra_maps(
        action: Action,
        current_maps: Dict[str, str],
        variable_map: Dict[str, str],
        **kwargs: Any,
    ) -> List[str]:
        def _add_to_list() -> None:
            mapped_item = current_maps.get(item, f'"${item}$"')
            referred_item = (
                variable_map.get(item, mapped_item)
                if mapped_item == f'"${item}$"'
                else mapped_item
            )

            temp_array.append(f"{item}={referred_item}" if referred_item else item)

        catalog = kwargs.get("catalog", Catalog())

        backing_sequence: SequencingData = kwargs.get("sequence", SequencingData())

        backing_step = next(
            (
                step
                for step in backing_sequence.output
                if step.name == action.name and step.arguments.keys() == action.inputs
            ),
            None,
        )

        temp_array: List[str] = list()

        if backing_step:
            cached_items: List[str] = list()

            backing_spec = catalog.get_api(name=backing_step.name)
            required_arguments = (
                backing_spec.get_arguments(required=True)
                if isinstance(backing_spec, API)
                else []
            )

            for item in backing_step.arguments:
                if item in action.inputs:
                    cached_items.append(item)
                    _add_to_list()

            for item in action.inputs:
                if item not in cached_items and item in required_arguments:
                    _add_to_list()

        else:
            for item in action.inputs:  # noqa
                _add_to_list()

        return temp_array

    @classmethod
    def print_out_action(
        cls,
        action: Action,
        current_maps: Dict[str, str],
        variable_map: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        collapse_maps: bool = kwargs.get("collapse_maps", False)

        if collapse_maps:
            temp_array = cls.add_extra_maps(
                action, current_maps, variable_map, **kwargs
            )
            input_string = ", ".join(temp_array) or None
        else:
            input_string = ", ".join(action.inputs) or None

        input_string = f"({input_string or ''})"
        output_variable = action.parameters[0] if action.parameters else ""

        if output_variable:
            for output in action.outputs:
                variable_map[output] = f'"${output_variable}.{output}$"'

        output_string = f"{output_variable} = " if output_variable else ""

        new_string = f"{output_string}{action.name}{input_string}"
        return new_string

    @classmethod
    def pretty_print_plan(cls, plan: Plan, **kwargs: Any) -> str:
        collapse_maps: bool = kwargs.get("collapse_maps", False)
        memory: Dict[str, Any] = kwargs.get("memory", {})

        current_index = 1
        current_maps: Dict[str, str] = cls.set_memory(memory)
        variable_map: Dict[str, str] = dict()
        pretty = []

        for action in plan.plan:
            which_basic = BasicOperations.which_basic(action.name)
            new_string = ""

            if which_basic:
                if which_basic == BasicOperations.CONSTRAINT:
                    raise NotImplementedError

                if which_basic == BasicOperations.SLOT_FILLER:
                    parameter = action.inputs[0]
                    new_string = f"{which_basic.value}({parameter})"

                elif which_basic == BasicOperations.CONFIRM:
                    key = action.inputs[0]
                    value = current_maps.get(key, key)

                    new_string = (
                        f"{which_basic.value}({key}={variable_map.get(value, value)})"
                    )

                elif which_basic == BasicOperations.MAPPER:
                    source = action.inputs[0]
                    target = action.inputs[1]

                    if len(action.inputs) == 3:
                        label = action.inputs[2]
                        if label == get_token_predicate_name(index=0, token="var"):
                            source = (
                                f'"${source}$"'
                                if not source.startswith('"')
                                else source
                            )

                        elif label != get_token_predicate_name(index=0, token="var"):
                            source = f'"${label}.{source}$"'

                    current_maps[target] = source

                    if not collapse_maps:
                        new_string = f"{which_basic.value}({source}, {target})"
            else:
                action.inputs = cls.filter_inputs(action, current_maps, **kwargs)

                new_string = cls.print_out_action(
                    action, current_maps, variable_map, **kwargs
                )

                current_index += 1

            if new_string:
                pretty.append(new_string)

        return "\n".join(pretty)

    @classmethod
    def parse_tokens(
        cls, list_of_tokens: List[str], **kwargs: Any
    ) -> ClassicalPlanReference:
        parsed_plan = ClassicalPlanReference()
        cached_maps: Dict[str, str] = dict()

        for token in list_of_tokens:
            new_action = cls.parse_token(token, cached_maps=cached_maps, **kwargs)

            if new_action:
                parsed_plan.plan.append(new_action)
            else:
                continue

        return parsed_plan

    @classmethod
    def parse_token(cls, token: str, **kwargs: Any) -> Union[Step, None]:
        cached_maps = kwargs.get("cached_maps", {})

        try:
            match_object = match(
                pattern=r"\s*(\[[0-9]+]\s+)?(?P<token>.*)\s*", string=token
            )
            token = (
                ""
                if match_object is None
                else match_object.groupdict().get("token", "")
            )
            token = token.strip()

            if token.startswith(BasicOperations.SLOT_FILLER.value):
                action_name, parameters = parse_parameters(token)
                new_action = Step(name=action_name, parameters=parameters)

            elif token.startswith(BasicOperations.CONFIRM.value):
                action_name, parameters = parse_parameters(token)
                new_action = Step(
                    name=action_name,
                    parameters=[cls.extract_variable(p) for p in parameters],
                )

            elif token.startswith(BasicOperations.MAPPER.value):
                action_name, parameters = parse_parameters(token)
                label, mapping = extract_label(parameters[0]) or MappingLabel(
                    label=get_token_predicate_name(index=0, token="var")
                )

                source = mapping or parameters[0]
                target = parameters[1]

                cached_maps[target] = source

                new_action = Step(
                    name=action_name,
                    label=label,
                    parameters=[source, target],
                )

            else:
                action_split = token.split("=")
                action_split = [item.strip() for item in action_split]

                label = None if len(action_split) == 1 else action_split[0]
                agent_signature_string = (
                    action_split[0]
                    if len(action_split) == 1
                    else "=".join(action_split[1:])
                )

                action_name, parameters = parse_parameters(agent_signature_string)

                parameters = [cls.extract_variable(p) for p in parameters]
                new_action = Step(
                    name=action_name,
                    label=label,
                    parameters=parameters,
                    maps=[cached_maps.get(p, p) for p in parameters],
                )

            return new_action

        except Exception as e:
            warn(
                message=f"Unrecognized token: {token}, {e}",
                category=SyntaxWarning,
                stacklevel=2,
            )
            return None

    @classmethod
    def extract_variable(cls, string: str) -> str:
        split = string.split("=")
        return split[0] if len(split) == 1 else split[1]

    @classmethod
    def filter_inputs(
        cls, action: Action, current_maps: Dict[str, str], **kwargs: Any
    ) -> List[str]:
        catalog = kwargs.get("catalog")
        sequence = kwargs.get("sequence")

        if not catalog:
            raise ValueError("Must provide catalog!")

        api_spec = catalog.get_api(name=action.name)
        assert api_spec is not None, "Refracted plan cannot contain made up APIs"

        required_arguments = api_spec.get_arguments(required=True)
        optional_arguments = api_spec.get_arguments(required=False)

        filtered_inputs = []

        for ip in action.inputs:
            if ip in required_arguments:
                filtered_inputs.append(ip)

            elif ip in optional_arguments:
                cached_parameter_names = []

                if sequence is not None:
                    for sequence_step in sequence.output:
                        if sequence_step.name == action.name:
                            cached_parameter_names.extend(
                                list(sequence_step.arguments.keys())
                            )

                if ip in cached_parameter_names or ip in current_maps.keys():
                    filtered_inputs.append(ip)
            else:
                raise ValueError(
                    f"Cannot have unknown parameter {ip} in refracted plan"
                )

        return filtered_inputs
