"""Intermediate task layer for image+text contrastive tasks.

Holds the CLIP-family modality contract: ``data_keys = ("image", "text")``,
positional ``task(image, text)`` backward-compat in ``forward()``,
``create_dummy_batch`` for FSDP eval scaffolding, and ``clamp_logit_scale``.
Concrete tasks (CLIPTask, SigLIPTask, CoCaTask, DistillCLIPTask) inherit
from this layer.

Future modalities (NaFlex, CLAP, MamMuT) should derive directly from
``TrainingTask`` and supply their own contract.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base_task import TrainingTask, unwrap_model


class ImageTextTask(TrainingTask):
    """Image + text contract shared by CLIP-family tasks."""

    @property
    def data_keys(self) -> Tuple[str, ...]:
        """Keys expected in the batch dict from the data pipeline."""
        return ("image", "text")

    def create_dummy_batch(
            self,
            image_size,
            context_length: int,
            batch_size: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create a dummy batch for FSDP eval on non-rank-0 workers."""
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)
        return {
            "image": torch.zeros(batch_size, 3, *image_size, device=device, dtype=dtype),
            "text": torch.zeros(batch_size, context_length, device=device, dtype=torch.long),
        }

    def clamp_logit_scale(self, max_val: float = math.log(100)):
        """Clamp logit_scale parameter to [0, max_val].

        With FSDP2, logit_scale is a replicated DTensor. In-place clamp_
        dispatches to the local tensor on each rank, which is correct for
        a single-element replicated param.
        """
        model = unwrap_model(self.trainable_module)
        if hasattr(model, 'logit_scale'):
            with torch.no_grad():
                model.logit_scale.clamp_(0, max_val)

    def forward(self, *args, **kwargs):
        """Accept dict, kwargs, positional ``(image, text)``, or mixed.

        Normalizes all calling conventions to a single dict before delegating
        to ``TrainingTask.forward(batch)``.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            batch = args[0]
            if kwargs:
                batch = {**batch, **kwargs}
        elif args and kwargs:
            batch = dict(zip(self.data_keys, args))
            batch.update(kwargs)
        elif args:
            batch = dict(zip(self.data_keys, args))
        else:
            batch = kwargs
        return super().forward(batch)
