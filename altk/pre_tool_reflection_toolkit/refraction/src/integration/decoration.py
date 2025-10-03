from typing import Dict, Any, Optional
from nl2flow.compile.options import BasicOperations
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import DebuggingResult
from nestful import SequenceStep
from nestful.memory import resolve_in_memory
from nestful.utils import get_token, TOKEN
from typing import Callable, TypeVar, ParamSpec, Concatenate
from functools import wraps

import inspect
import sys

P = ParamSpec("P")
R = TypeVar("R")


def generate_free_label(memory: Dict[str, Any], max_token: int = 100) -> Optional[str]:
    used_keys = {key for key in memory.keys() if key.startswith(TOKEN)}

    for index in range(1, max_token):
        tmp_token = get_token(index)

        if tmp_token not in used_keys:
            return str(tmp_token)

    return None


def refract(
    api: Optional[str] = None,
    use_given_operators_only: bool = True,
    use_cc: bool = False,
    allow_remaps: bool = False,
    execute_if_fixed: bool = False,
    use_state: bool = False,
) -> Callable[
    [Callable[P, R | DebuggingResult]],
    Callable[Concatenate[Refractor, Optional[Dict[str, Any]], P], R | DebuggingResult],
]:
    def refract_wrapper(
        function: Callable[P, R],
    ) -> Callable[
        Concatenate[Refractor, Optional[Dict[str, Any]], P], R | DebuggingResult
    ]:
        @wraps(function)
        def run_main(
            refractor: Optional[Refractor | Dict[str, Any]] = None,
            memory: Optional[Dict[str, Any]] = None,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R | DebuggingResult:
            memory = memory or {}
            label = generate_free_label(memory)

            current_name = api or function.__name__

            if isinstance(refractor, Dict):
                kwargs = {**refractor}  # type: ignore

                refractor = refractor.get("refractor", None)
                memory = kwargs.get("memory", {})  # type: ignore

                del kwargs["refractor"]
                del kwargs["memory"]

                memory = {**memory, **kwargs}

            if isinstance(refractor, Refractor):
                sequence_object = SequenceStep(
                    name=current_name,
                    arguments=kwargs,
                    label=label,
                )

                result: DebuggingResult = refractor.refract(
                    sequence=sequence_object,
                    memory_objects=memory,
                    use_given_operators_only=use_given_operators_only,
                    use_cc=use_cc,
                    allow_remaps=allow_remaps,
                )
            else:
                # TODO: For now letting it pass
                result = DebuggingResult()
                result.report.determination = True

            if result.report.determination:
                return (
                    function(*args, **kwargs)  # type: ignore
                    if use_state is False
                    else function(state=kwargs)  # type: ignore
                )

            else:
                cfc = result.corrected_function_call(memory, catalog=refractor.catalog)

                if cfc is None:
                    return result

                if cfc.is_executable and execute_if_fixed:
                    module_name = function.__module__

                    if module_name is None:
                        raise ModuleNotFoundError(
                            f"Tried to execute: {function.__name__}"
                        )

                    try:
                        module = sys.modules[module_name]
                    except KeyError:
                        raise KeyError(f"Couldn't load module {module_name}") from None

                    current_index = 0

                    for index, item in enumerate(cfc.objectified.output):
                        function_name = item.name

                        if function_name == current_name:
                            current_index = index
                            break

                        else:
                            for name, member in inspect.getmembers(module):
                                if inspect.isfunction(member) and name == function_name:
                                    resolved_arguments = resolve_in_memory(
                                        item.arguments, memory
                                    )
                                    item.arguments = resolved_arguments

                                    pretty = item.pretty_print(
                                        mapper_tag=BasicOperations.MAPPER.value,
                                        collapse_maps=True,
                                    )

                                    print(f"Executing: {pretty}")

                                    output = member(
                                        refractor=refractor,
                                        memory=memory,
                                        **resolved_arguments,
                                    )
                                    memory[item.label] = output

                    resolved_arguments = resolve_in_memory(
                        cfc.objectified.output[current_index].arguments, memory
                    )

                    new_kwargs = {
                        **kwargs,
                        **resolved_arguments,
                    }

                    pretty = SequenceStep(
                        name=current_name, arguments=new_kwargs
                    ).pretty_print(
                        mapper_tag=BasicOperations.MAPPER.value,
                        collapse_maps=True,
                    )

                    print(f"Executing: {pretty}")
                    return (
                        function(*args, **new_kwargs)
                        if use_state is False
                        else function(state=new_kwargs)  # type: ignore
                    )

                else:
                    return result

        return run_main

    return refract_wrapper
