"""
Evaluation helpers for alignment model.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .encoders_text import tokenize_batch


@torch.no_grad()
def eval_alignment(model, dl, tok, clip_image_emb_fn, clip_text_emb_fn, device):
    model.eval()
    stats = {"cos_predimg_tgtimg": [], "cos_predimg_txtemb": [], "txt_loss": []}
    pbar = tqdm(dl, desc="eval")
    for batch in pbar:
        batch["ctx_images"] = batch["ctx_images"].to(device)
        batch["tgt_image"] = batch["tgt_image"].to(device)

        z = model.encoder(batch["ctx_images"], batch["ctx_captions"])
        enc, _ = tokenize_batch(tok, batch["ctx_captions"], batch["tgt_caption"])
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        labels = enc["labels"].to(device)

        txt_loss, txt_emb, _ = model.textdec(input_ids, attn_mask, labels, z)
        tgt_img_emb = clip_image_emb_fn(batch["tgt_image"])
        pred_img_emb = torch.nn.functional.normalize(model.img_head(z), dim=-1)

        cos1 = (pred_img_emb * tgt_img_emb).sum(dim=-1)
        cos2 = (pred_img_emb * txt_emb).sum(dim=-1)

        stats["cos_predimg_tgtimg"] += cos1.detach().cpu().tolist()
        stats["cos_predimg_txtemb"] += cos2.detach().cpu().tolist()
        stats["txt_loss"].append(float(txt_loss.detach().cpu()))

        pbar.set_postfix(
            cos_predimg_tgtimg=np.mean(stats["cos_predimg_tgtimg"][-64:]) if stats["cos_predimg_tgtimg"] else 0.0,
            cos_predimg_txt=np.mean(stats["cos_predimg_txtemb"][-64:]) if stats["cos_predimg_txtemb"] else 0.0,
        )

    summary = {k: (float(np.mean(v)) if len(v) else None) for k, v in stats.items()}
    return summary
