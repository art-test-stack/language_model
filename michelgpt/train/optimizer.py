from michelgpt.settings import *

import torch
from torch import Tensor, optim
from typing import Iterable, Dict, Any, Tuple

class AdamW(optim.AdamW):
    def __init__(
            self,
            params: Iterable[Tensor] | Iterable[Dict[str, Any]],
            lr: float | Tensor = LEARNING_RATE,
            eps: float = EPSILON,
            weight_decay: float = WEIGHT_DECAY,
            fused: bool | None = None
        ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=(BETA_1, BETA_2),
            eps=eps,
            fused=fused
        )

    def update_learning_rate(self):
        if self.step < WARMUP_STEPS:
            ratio = self.step / WARMUP_STEPS
            self.learning_rate = MAX_LEARNING_RATE * ratio
		
        elif self.step < WARMUP_STEPS + DECAY_STEPS:
            ratio = (self.step - WARMUP_STEPS) / DECAY_STEPS
            ratio = 0.5 * (1.0 + torch.cos(torch.pi * ratio))
            self.learning_rate = ratio * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) + MIN_LEARNING_RATE
		
        else:
            self.learning_rate = MIN_LEARNING_RATE
        
        for g in self.param_groups:
            g['lr'] = self.learning_rate