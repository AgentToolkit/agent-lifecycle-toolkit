import logging
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from altk.core.toolkit import AgentPhase, ComponentBase
from pydantic import ConfigDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from altk.pre_llm.core.config import (
    SpotLightConfig,
    SpotLightMetadata,
    SpotLightOutputSchema,
    SpotLightRunInput,
    SpotLightRunOutput,
)

warnings.simplefilter("ignore", UserWarning)

logger = logging.getLogger(__name__)


class SpotLightComponent(ComponentBase):
    """
    Component to improve LLM instruction following using SpotLight's attention steering.

    It enables users to emphasize spans within their prompts that they want the LLM to pay more attention to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SpotLightConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    generation_kwargs: Dict
    num_attn_heads: int
    num_layers: int

    def __init__(self, config: Optional[SpotLightConfig] = None, **kwargs):
        if not config:
            logger.warning("No config given to SpotLight. Using default.")
        config_obj = config or SpotLightConfig()
        model, tokenizer, num_heads, num_layers, generation_kwargs = (
            self._load_model_components(config_obj)
        )
        data = {
            "config": config_obj,
            "model": model,
            "tokenizer": tokenizer,
            "num_attn_heads": num_heads,
            "num_layers": num_layers,
            "generation_kwargs": generation_kwargs,
        }
        data.update(kwargs)
        super().__init__(**data)
        logger.info("Successfully initialized SpotLight middleware")

    def _load_model_components(
        self, config: SpotLightConfig
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, int, int, Dict]:
        """Initialize SpotLight variables."""
        try:
            generation_kwargs = config.generation_kwargs
            generation_kwargs.update(
                {"output_attentions": True}
            )  # Mandatory for spotlight
            device_config = self.get_device_config(generation_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path, **device_config
            )
            tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            num_attn_heads = model.config.num_attention_heads
            num_layers = model.config.num_hidden_layers

            return model, tokenizer, num_attn_heads, num_layers, generation_kwargs

        except Exception as e:
            error_msg = f"Issue loading HF model for spotlight: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_device_config(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract device configuration from generation_kwargs.

        Priority: device > device_map > default to 'auto'
        """
        if "device" in generation_kwargs and "device_map" in generation_kwargs:
            logger.warning(
                "Both 'device' and 'device_map' specified in generation_kwargs. "
                "Using 'device' and ignoring 'device_map'."
            )
            return {"device": generation_kwargs["device"]}
        elif "device" in generation_kwargs:
            return {"device": generation_kwargs["device"]}
        elif "device_map" in generation_kwargs:
            return {"device_map": generation_kwargs["device_map"]}
        else:
            return {"device_map": "auto"}

    def tokenize_inputs(
        self, raw_inputs: list, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[BatchEncoding, Tuple, int]:
        """Tokenize inputs and obtain their offset mappings."""
        _padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            raw_inputs, return_tensors="pt", return_offsets_mapping=True, padding=True
        )
        offset_mappings = inputs.pop("offset_mapping")
        tokenized_input_lengths = inputs["input_ids"].shape[1]
        tokenizer.padding_side = _padding_side
        return inputs, offset_mappings, tokenized_input_lengths

    def find_span(
        self,
        prompt: str,
        emph_string: str,
        offset_mapping: Sequence[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Get start and end indices of given span within the prompt"""
        if offset_mapping is None:
            raise ValueError("must provide offset_mapping")
        if emph_string not in prompt:
            raise ValueError(f'"{emph_string}" not found in "{prompt}"')

        start_idx = prompt.find(emph_string)
        end_idx = start_idx + len(emph_string)

        span_start, span_end = None, None
        for index, (tk_start, tk_end) in enumerate(offset_mapping):
            if span_start is None:
                if tk_start <= start_idx and tk_end >= start_idx:
                    span_start = index
            if span_end is None:
                if tk_start <= end_idx and tk_end >= end_idx:
                    span_end = index
                    break
        assert (
            span_start is not None and span_end is not None and span_start <= span_end
        )
        return (span_start, span_end + 1)

    def get_span_range(
        self,
        prompts: List[str],
        emph_strings: List[str] | List[List[str]],
        offset_mappings: Sequence[Sequence[Tuple[int, int]]],
    ):
        """Find the start and end indices of the spans to emphasize within the prompt
        Args:
            prompts: Input prompt(s)
            emph_strings: Span(s) to emphasize within the prompt
            offset_mappings: Tokenizer offset mappings

        Returns: List of tuples specifying the start and end indices of the spans
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if len(emph_strings) == 0:
            emph_strings = []
        elif isinstance(emph_strings[0], str):
            emph_strings = [[s] for s in emph_strings]

        assert len(prompts) == len(emph_strings), (
            "Mismatch prompts ↔ emphasised strings"
        )

        span_ranges_per_sample = []
        for prompt, span_list, offsets in zip(prompts, emph_strings, offset_mappings):
            sample_ranges = []
            for span in span_list:
                if not span:
                    continue
                if span in prompt:
                    if prompt.count(span) != 1:
                        logger.warning(f"Ambiguous span {span}")
                    try:
                        rng = self.find_span(prompt, span, offset_mapping=offsets)
                        sample_ranges.append(rng)
                    except ValueError as e:
                        logger.error(f"Cannot find span: {str(e)}")
                        raise
                if span not in prompt:
                    logger.error(f"Cannot find span {span} in {prompt}")
                    raise
            span_ranges_per_sample.append(sample_ranges)
        return span_ranges_per_sample

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """KV-cache helper function to recompute attention outputs."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def create_attention_bias_hook(self, span_ranges: List, target_proportion: float):
        """SpotLight's forward hook to modify attention weights
        Args:
            span_ranges: List of span ranges
            target_proportion: Target attention proportion to emphasize span towards
        """

        def attention_bias_hook(module, input_args, input_kwargs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            attn_output, attn_weights = output[0], output[1]
            del attn_output
            modified_weights = attn_weights.clone()

            for i, ranges in enumerate(span_ranges):
                if not ranges:
                    continue

                union_mask = torch.zeros(
                    modified_weights.size(-1),
                    device=modified_weights.device,
                    dtype=modified_weights.dtype,
                )
                for start, end in ranges:
                    union_mask[start:end] = 1.0
                union_mask = union_mask.view(1, 1, -1)

                current_proportion = (
                    modified_weights[i] * union_mask
                ).sum() / modified_weights[i].sum()

                if current_proportion < target_proportion:
                    bias_value = torch.log(
                        torch.tensor(
                            target_proportion / current_proportion,
                            device=modified_weights.device,
                        )
                    )
                    bias_mask = union_mask * bias_value
                    attn_logits = torch.log(modified_weights[i] + 1e-10)
                    attn_logits += bias_mask
                    attn_logits = F.softmax(
                        attn_logits, dim=-1, dtype=torch.float32
                    ).to(modified_weights.dtype)
                    modified_weights[i] = attn_logits

            _, value_states = input_kwargs["past_key_value"][module.layer_idx]
            if value_states.shape[1] == input_kwargs["hidden_states"].shape[1]:
                value_states = value_states.view(
                    value_states.shape[0],
                    value_states.shape[1],
                    module.config.num_key_value_heads,
                    module.config.head_dim,
                ).transpose(1, 2)
            if module.config.num_attention_heads != module.config.num_key_value_heads:
                value_states = self.repeat_kv(value_states, module.num_key_value_groups)

            new_attn_output = torch.matmul(modified_weights, value_states)
            batch_size, _, seq_len = (
                new_attn_output.size(0),
                new_attn_output.size(1),
                new_attn_output.size(2),
            )
            new_attn_output = (
                new_attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, -1)
            )
            new_attn_output = module.o_proj(new_attn_output)
            return (new_attn_output, None, input_kwargs["past_key_value"])

        return attention_bias_hook

    @contextmanager
    def steer(
        self,
        model: PreTrainedModel,
        prompts: list,
        emph_strings: list,
        offset_mappings: Sequence[Sequence[tuple[int, int]]],
        target_proportion: float,
    ):
        registered_hooks = []
        span_ranges = []

        # Check if steering is needed
        if (
            emph_strings is not None
            and len(emph_strings) > 0
            and any(s != "" for s in emph_strings)
        ):
            span_ranges = self.get_span_range(prompts, emph_strings, offset_mappings)
            assert len(span_ranges) == len(prompts), "1 span-list per sample expected"

            for layer_idx in range(self.num_layers):
                name = f"model.layers.{layer_idx}.self_attn"
                module = model.get_submodule(name)
                hook_func = self.create_attention_bias_hook(
                    span_ranges, target_proportion
                )
                registered_hook = module.register_forward_hook(
                    hook_func, with_kwargs=True
                )
                registered_hooks.append(registered_hook)
        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """Return the supported agent phases."""
        return {AgentPhase.RUNTIME}

    def _run(self, data: SpotLightRunInput) -> SpotLightRunOutput:
        try:
            if isinstance(data.metadata, SpotLightMetadata):
                spotlight_metadata = data.metadata
            elif data.metadata and isinstance(data.metadata, dict):
                spotlight_metadata = SpotLightMetadata(**data.metadata)
            else:
                spotlight_metadata = SpotLightMetadata()
            target_proportion = spotlight_metadata.alpha
            emph_strings = spotlight_metadata.emph_strings
            emph_strings = (
                emph_strings if isinstance(emph_strings, list) else [emph_strings]
            )

            messages = data.messages

            chat = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            ]
            assert len(emph_strings) == len(chat), (
                "Need 1:1 mapping between number of prompts and emph_strings per prompt"
            )

            inputs, offset_mappings, tokenized_inp_len = self.tokenize_inputs(
                chat, self.tokenizer
            )

            with self.steer(
                model=self.model,
                prompts=chat,
                emph_strings=emph_strings,
                offset_mappings=offset_mappings,
                target_proportion=target_proportion,
            ) as spotlight_model:
                outputs = spotlight_model.generate(
                    **inputs.to(self.model.device), **self.generation_kwargs
                )
            decoded_output = self.tokenizer.decode(
                outputs[0][tokenized_inp_len:], skip_special_tokens=True
            )
            return SpotLightRunOutput(
                output=SpotLightOutputSchema(
                    prediction=decoded_output, metadata=spotlight_metadata
                )
            )

        except Exception as e:
            logger.error(f"SpotLight run failed: {e}.")
            return SpotLightRunOutput(output=SpotLightOutputSchema(prediction="Error."))
