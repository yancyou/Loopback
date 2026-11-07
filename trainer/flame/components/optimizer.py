from __future__ import annotations

from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer, build_optimizers as build_torchtitan_optimizers
from torchtitan.tools.logging import logger

T = TypeVar("T", bound=Optimizer)


def _is_token_embedding(name: str) -> bool:
    # the input embedding (`embeddings.weight`); never lm_head, which is untied
    return name.endswith("embeddings.weight") and "lm_head" not in name


# optimizer.wd_params: which parameters weight decay applies to.
#   all                every parameter (torchtitan's behaviour; the paper's runs, with weight decay 0)
#   matrices           parameters with ndim >= 2 only (the nanoGPT / GPT-NeoX rule): RMSNorm gains, biases
#                      and every 1-D parameter are not decayed; embeddings, lm_head and all weight
#                      matrices are
#   matrices_no_embed  as `matrices`, and the token embedding is not decayed either (OLMo 2)
WD_PARAMS_MODES = ("all", "matrices", "matrices_no_embed")


def _decayed(name: str, param: nn.Parameter, wd_params: str) -> bool:
    if wd_params == "all":
        return True
    if param.ndim < 2:
        return False
    return not (wd_params == "matrices_no_embed" and _is_token_embedding(name))


def _param_groups(model: nn.Module, wd_params: str) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    if wd_params not in WD_PARAMS_MODES:
        raise ValueError(f"optimizer.wd_params must be one of {WD_PARAMS_MODES}, got {wd_params!r}")
    decay, no_decay, all_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        all_params.append(param)
        (decay if _decayed(name, param, wd_params) else no_decay).append(param)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups, all_params


class GroupedOptimizersContainer(OptimizersContainer[T]):
    """torchtitan's OptimizersContainer with one decayed and one undecayed parameter group per model part."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        wd_params: str,
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        weight_decay = optimizer_kwargs["weight_decay"]
        for model in self.model_parts:
            groups, params = _param_groups(model, wd_params)
            n_decay = sum(p.numel() for g in groups if g.get("weight_decay", weight_decay) != 0.0 for p in g["params"])
            n_all = sum(p.numel() for p in params)
            logger.info(f"weight decay {weight_decay} (wd_params={wd_params}) on {n_decay:,} of {n_all:,} parameters "
                        f"({len(groups)} groups: " + ", ".join(f"{len(g['params'])} tensors wd={g.get('weight_decay', weight_decay)}"
                                                               for g in groups) + ")")
            self.optimizers.append(optimizer_cls(groups, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    job_config,
    ft_manager: FTManager,
):
    wd_params = getattr(job_config.optimizer, "wd_params", "all") or "all"
    if wd_params not in WD_PARAMS_MODES:
        raise ValueError(f"optimizer.wd_params must be one of {WD_PARAMS_MODES}, got {wd_params!r}")
    if wd_params == "all":
        return build_torchtitan_optimizers(model_parts, job_config, ft_manager)

    if job_config.optimizer.early_step_in_backward or ft_manager.enabled:
        raise NotImplementedError("optimizer.wd_params other than 'all' is only implemented for the plain optimizer container")
    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if job_config.optimizer.name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {job_config.optimizer.name} not added.")
    optimizer_kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
        "weight_decay": job_config.optimizer.weight_decay,
        "fused": job_config.optimizer.implementation == "fused",
        "foreach": job_config.optimizer.implementation == "foreach",
    }
    return GroupedOptimizersContainer(model_parts, optimizer_classes[job_config.optimizer.name], optimizer_kwargs, wd_params)
