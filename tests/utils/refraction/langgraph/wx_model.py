from typing import Any, Dict, Union, List, Optional, Callable, Sequence
from tests.utils.refraction.langgraph.utils import (
    generate_prompt,
    extract_tool_calls,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelInput

import os
import requests  # type: ignore


class WXModel(BaseChatModel):
    # NOTE: This class needs to be refactored to use WX or some other LLM provider
    model_name: str
    model_url: str
    tools: List[Dict[str, Any]] = []

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        model_url = f"{self.model_url}/v1/chat/completions"

        headers: dict[str, str] = {
            "content-type": "application/json",
            "WX_API_KEY": str(os.environ.get("WX_API_KEY")),
        }

        tools = [item["function"] for item in self.tools]
        prompt = generate_prompt(query=messages[0].content, list_of_tools=tools)

        messages = [{"role": "user", "content": prompt}]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": messages,
            "seed": 16,
        }

        response = requests.post(
            model_url, headers=headers, json=payload, verify=False
        ).json()

        model_response = response["choices"][0]["message"]["content"]
        tool_calls = extract_tool_calls(model_response)

        message = AIMessage(
            content=model_response,
            additional_kwargs=response,
            response_metadata={
                "model_name": response["model"],
            },
            usage_metadata={
                "input_tokens": response["usage"]["prompt_tokens"],
                "output_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"],
            },
            tool_calls=[
                ToolCall(id=str(index), **tc) for index, tc in enumerate(tool_calls)
            ],
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable[..., Any], BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        self.tools = formatted_tools
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name
