"""
Alignment loss utilities (CLIP-based).
"""
from __future__ import annotations

import torch


def alignment_loss(pred_img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
    return (1.0 - (pred_img_emb * txt_emb).sum(dim=-1)).mean()
