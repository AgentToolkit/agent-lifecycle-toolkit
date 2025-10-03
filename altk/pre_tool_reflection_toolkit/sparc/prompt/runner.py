import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union, Tuple
from pydantic import BaseModel

Prompt = Union[str, List[Dict[str, Any]]]
PromptAndSchema = Tuple[Union[str, List[Dict[str, Any]]], Optional[Dict[str, Any]]]
SyncGen = Callable[[Prompt], Union[str, Any]]
BatchGen = Callable[[List[Prompt]], List[Union[str, Any]]]
AsyncGen = Callable[[Prompt], Awaitable[Union[str, Any]]]
AsyncBatchGen = Callable[[List[Prompt]], Awaitable[List[Union[str, Any]]]]

T = TypeVar("T")


class PromptResult(BaseModel):
    """
    Holds the prompt sent and the response (or error).
    """

    prompt: Prompt
    response: Optional[Any] = None
    error: Optional[str] = None


class PromptRunner:
    """
    Runs a collection of prompts through various generation strategies.

    Attributes:
        prompts: the list of prompts to run.
    """

    def __init__(
        self, prompts: Optional[List[Union[Prompt, PromptAndSchema]]] = None
    ) -> None:
        """
        Args:
            prompts: initial list of prompts (strings or chat messages).
        """
        self.prompts: List[Union[Prompt, PromptAndSchema]] = prompts or []

    def add_prompt(self, prompt: Union[Prompt, PromptAndSchema]) -> None:
        """Append a prompt to the runner."""
        self.prompts.append(prompt)

    def remove_prompt(self, prompt: Union[Prompt, PromptAndSchema]) -> None:
        """Remove a prompt (first occurrence)."""
        self.prompts.remove(prompt)

    def clear_prompts(self) -> None:
        """Remove all prompts."""
        self.prompts.clear()

    def get_prompt_and_schema(
        self, prompt: Union[Prompt, PromptAndSchema]
    ) -> Tuple[Prompt, Optional[Dict[str, Any]]]:
        """
        Extract the prompt and schema from a Prompt object.

        Args:
            prompt: The prompt to extract from.

        Returns:
            Tuple of (prompt, schema).
        """
        if isinstance(prompt, tuple):
            return prompt[0], prompt[1]
        return prompt, None

    def run_all(
        self,
        gen_fn: SyncGen,
        prompt_param_name: str = "prompt",
        schema_param_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[PromptResult]:
        """
        Run each prompt through a synchronous single-prompt generator.

        Args:
            gen_fn: Callable taking one Prompt, returning str or Any.
            prompt_param_name: Name of the parameter for the prompt.
            schema_param_name: Name of the parameter for the schema.
            kwargs: Additional arguments to pass to the function.

        Returns:
            List of PromptResult.
        """
        results: List[PromptResult] = []
        for p in self.prompts:
            try:
                prompt, schema = self.get_prompt_and_schema(p)
                args = {prompt_param_name: prompt, **kwargs}
                if schema_param_name and schema:
                    args[schema_param_name] = schema
                resp = gen_fn(**args)
                results.append(PromptResult(prompt=prompt, response=resp))
            except Exception as e:
                results.append(PromptResult(prompt=prompt, error=str(e)))
        return results

    async def run_async(
        self,
        async_fn: AsyncGen,
        max_parallel: int = 10,
        prompt_param_name: str = "prompt",
        schema_param_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[PromptResult]:
        """
        Run each prompt through an async single-prompt generator with concurrency limit.
        Results are returned in the same order as self.prompts.

        Args:
            async_fn: Async callable taking one Prompt, returning str or Any.
            max_parallel: Max concurrent tasks.
            prompt_param_name: Name of the parameter for the prompt.
            schema_param_name: Name of the parameter for the schema.
            kwargs: Additional arguments to pass to the async function.

        Returns:
            List of PromptResult.
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_one(index: int, p: Prompt) -> Tuple[int, PromptResult]:
            async with semaphore:
                try:
                    prompt, schema = self.get_prompt_and_schema(p)
                    args = {prompt_param_name: prompt, **kwargs}
                    if schema_param_name and schema:
                        args[schema_param_name] = schema
                    resp = await async_fn(**args)
                    return index, PromptResult(prompt=prompt, response=resp)
                except Exception as e:
                    return index, PromptResult(prompt=prompt, error=str(e))

        tasks = [
            asyncio.create_task(_run_one(i, p)) for i, p in enumerate(self.prompts)
        ]
        indexed_results = await asyncio.gather(*tasks)
        # Sort results to match original order
        indexed_results.sort(key=lambda x: x[0])
        return [res for _, res in indexed_results]
