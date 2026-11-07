from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_looped_transformer import LoopedTransformerConfig
from .modeling_looped_transformer import LoopedTransformerForCausalLM, LoopedTransformerModel

AutoConfig.register(LoopedTransformerConfig.model_type, LoopedTransformerConfig, exist_ok=True)
AutoModel.register(LoopedTransformerConfig, LoopedTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(LoopedTransformerConfig, LoopedTransformerForCausalLM, exist_ok=True)

__all__ = ["LoopedTransformerConfig", "LoopedTransformerForCausalLM", "LoopedTransformerModel"]
