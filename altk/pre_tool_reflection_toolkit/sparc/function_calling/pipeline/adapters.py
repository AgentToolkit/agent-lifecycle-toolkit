from typing import (
    Any,
    Dict,
    List,
)

from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.types import (
    ToolCall,
    ToolSpec,
)


# ────────────────────────────────────────────────────────────────────────────────
# Adapter definitions
# ────────────────────────────────────────────────────────────────────────────────


class BaseAdapter:
    """Abstract adapter to unify different API spec and call representations."""

    def get_tools_inventory(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_tools_inventory_summary(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_tool_spec(self, tool_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_call_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_function_name(self) -> str:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_param_spec_snippet(self, param_name: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    """Adapter for ToolSpec + ToolCall inputs."""

    def __init__(self, specs: List[ToolSpec], call: ToolCall):
        self.specs = specs
        self.call = call

    def get_tools_inventory(self) -> List[Dict[str, Any]]:
        return [spec.model_dump() for spec in self.specs]

    def get_tools_inventory_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool_name": spec.function.name,
                "tool_description": spec.function.description,
                "tool_parameters": {
                    prop_name: prop_d["type"]
                    for prop_name, prop_d in spec.function.parameters.get(
                        "properties", {}
                    ).items()
                },
            }
            for spec in self.specs
        ]

    def get_tool_spec(self, tool_name: str) -> Dict[str, Any]:
        tool = next((t for t in self.specs if t.function.name == tool_name), None)
        return tool.function.model_dump() if tool else {}

    def get_call_dict(self) -> Dict[str, Any]:
        call_dict = {
            "id": self.call.id,
            "type": "function",
            "function": {
                "name": self.call.function.name,
                "arguments": self.call.function.arguments,
            },
        }
        return call_dict

    def get_function_name(self) -> str:
        return self.call.function.name

    def get_parameters(self) -> Dict[str, Any]:
        return self.call.function.parsed_arguments

    def get_param_spec_snippet(self, param_name: str) -> Dict[str, Any]:
        spec = next(
            (s for s in self.specs if s.function.name == self.get_function_name()), None
        )
        if not spec:
            return {"type": "object", "properties": {}, "required": []}
        props = spec.function.parameters.get("properties", spec.function.parameters)
        if param_name not in props:
            return {"type": "object", "properties": {}, "required": []}
        return {
            "type": "object",
            "properties": {param_name: props[param_name]},
            "required": [param_name],
        }
