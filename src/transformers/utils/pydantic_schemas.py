# Copyright 2026 The HuggingFace Team / 8b.IS. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pydantic v2 Schemas & Fast-Path Dataclass Hybrid Architecture for Transformers-Ultra.
Provides Rust-accelerated schema validation at ingress/config boundaries combined with
zero-overhead slotted dataclasses for inner-loop forward execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    _has_pydantic = True
except ImportError:
    _has_pydantic = False
    BaseModel = object  # type: ignore


if _has_pydantic:

    class UltraGenerationConfigSchema(BaseModel):
        """Rust-accelerated Pydantic v2 schema for validating generation parameters."""

        model_config = ConfigDict(extra="allow", validate_assignment=True, populate_by_name=True)

        max_new_tokens: int = Field(default=512, gt=0, description="Maximum number of tokens to generate.")
        min_new_tokens: int = Field(default=0, ge=0, description="Minimum number of tokens to generate.")
        temperature: float = Field(default=1.0, ge=0.0, le=5.0, description="Sampling temperature.")
        top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Nucleus sampling probability threshold.")
        top_k: int = Field(default=50, ge=0, description="Top-k filtering threshold.")
        repetition_penalty: float = Field(default=1.0, gt=0.0, description="Repetition penalty parameter.")
        do_sample: bool = Field(default=False, description="Whether to use sampling or greedy decoding.")
        num_beams: int = Field(default=1, ge=1, description="Number of beams for beam search.")
        pad_token_id: int | None = Field(default=None, description="Padding token ID.")
        eos_token_id: int | list[int] | None = Field(default=None, description="End of sequence token ID.")

        @field_validator("temperature")
        @classmethod
        def validate_temperature_sampling(cls, v: float) -> float:
            if v == 0.0:
                # Temperature 0 is mathematically greedy decoding
                return v
            return v

        def to_dict(self) -> dict[str, Any]:
            """Converts valid schema directly to generation config dictionary."""
            return self.model_dump(exclude_none=True)

    class UltraQuantizationConfigSchema(BaseModel):
        """Pydantic v2 schema for quantization configuration validation.

        Per-method constraints keep nonsense configs from ever reaching a
        loader kernel: bitnets are 1-2 bit ternary, awq/gptq live at
        2/3/4/8 bits, fp8 is 8-bit float weights, and group sizes are
        powers of two (8..) whenever the method uses grouping. The method
        set is the modern transformers surface — fp8 ships DeepSeek's
        configs; rejecting it would make a real model unloadable.
        """

        model_config = ConfigDict(extra="forbid", validate_assignment=True)

        quant_method: Literal[
            "metal",
            "bitnet",
            "awq",
            "gptq",
            "bnb_4bit",
            "bnb_8bit",
            "compressed_tensors",
            "fp8",
            "hqq",
            "torchao",
            "eetq",
        ]
        bits: int = Field(default=4, ge=1, le=16)
        group_size: int | None = Field(default=None, ge=8)
        use_sub_norms: bool = Field(default=False)

        @field_validator("group_size")
        @classmethod
        def _group_size_power_of_two(cls, v):
            if v is not None and (v & (v - 1)) != 0:
                raise ValueError(f"group_size must be a power of two, got {v}")
            return v

        @model_validator(mode="after")
        def _check_method_bits(self):
            allowed = {
                "bitnet": {1, 2},  # 1.58-bit ternary, integer-coded
                "awq": {2, 3, 4, 8},
                "gptq": {2, 3, 4, 8},
                "bnb_4bit": {4},
                "bnb_8bit": {8},
                "fp8": {8},
                "compressed_tensors": {4, 8},
            }.get(self.quant_method)
            if allowed is not None and self.bits not in allowed:
                raise ValueError(
                    f"{self.quant_method} does not support bits={self.bits} "
                    f"(allowed {sorted(allowed)})"
                )
            return self

        def to_dict(self) -> dict[str, Any]:
            """Converts valid schema directly to a quantization config dict."""
            return self.model_dump(exclude_none=True)


# =====================================================================
# Fast-Path Zero-Overhead Slotted Dataclasses for Inner Forward Loops
# =====================================================================


@dataclass(slots=True, kw_only=True)
class UltraFastCausalLMOutput:
    """
    High-speed, zero-overhead slotted output container for generative models.
    Supports structural pattern matching:
        match output:
            case UltraFastCausalLMOutput(loss=l, logits=logits) if l is not None: ...
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor
    past_key_values: tuple[Any, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    def to_tuple(self) -> tuple[Any, ...]:
        return (self.loss, self.logits, self.past_key_values, self.hidden_states, self.attentions)
