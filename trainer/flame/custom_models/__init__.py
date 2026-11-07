"""Model registry of the exported trainer: the looped transformer only (importing it registers model_type
looped_transformer with transformers' Auto classes)."""

from .looped_transformer import LoopedTransformerConfig, LoopedTransformerForCausalLM, LoopedTransformerModel

__all__ = ["LoopedTransformerConfig", "LoopedTransformerForCausalLM", "LoopedTransformerModel"]
