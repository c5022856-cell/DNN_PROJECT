"""
Shared encoder + GPT-2 prefix conditioning model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders_text import format_prompt


class SharedEncoder(nn.Module):
    def __init__(self, clip_image_emb_fn, clip_text_emb_fn, d_model: int = 256, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.clip_image_emb_fn = clip_image_emb_fn
        self.clip_text_emb_fn = clip_text_emb_fn
        self.fuse = nn.Sequential(
            nn.Linear(512 + 512, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, ctx_images_bkchw, ctx_captions_list):
        B, K, C, H, W = ctx_images_bkchw.shape
        imgs = ctx_images_bkchw.reshape(B * K, C, H, W)
        with torch.no_grad():
            e_img = self.clip_image_emb_fn(imgs)
        flat_caps = []
        for caps in ctx_captions_list:
            for c in caps:
                flat_caps.append(c)
        with torch.no_grad():
            e_txt = self.clip_text_emb_fn(flat_caps)
        h = self.fuse(torch.cat([e_img, e_txt], dim=-1))
        h = h.reshape(B, K, -1)
        h = self.temporal(h)
        return h.mean(dim=1)


class PrefixGPT2(nn.Module):
    def __init__(self, gpt_model, d_z: int = 256, n_prefix: int = 8):
        super().__init__()
        self.gpt = gpt_model
        self.n_prefix = n_prefix
        self.z_to_prefix = nn.Linear(d_z, n_prefix * self.gpt.config.n_embd)
        self.txt_proj = nn.Linear(self.gpt.config.n_embd, 512)

    def forward(self, input_ids, attention_mask, labels, z):
        B, T = input_ids.shape
        tok_emb = self.gpt.transformer.wte(input_ids)
        prefix = self.z_to_prefix(z).view(B, self.n_prefix, -1)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        prefix_mask = torch.ones((B, self.n_prefix), dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)
        prefix_labels = torch.full((B, self.n_prefix), -100, dtype=labels.dtype, device=labels.device)
        lab = torch.cat([prefix_labels, labels], dim=1)

        out = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=lab,
            output_hidden_states=True,
            use_cache=False,
        )
        text_loss = out.loss
        hs = out.hidden_states[-1]
        mask = (lab != -100).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (hs * mask).sum(dim=1) / denom
        txt_emb = F.normalize(self.txt_proj(pooled), dim=-1)
        return text_loss, txt_emb, out


class AlignmentModel(nn.Module):
    def __init__(self, encoder: SharedEncoder, textdec: PrefixGPT2, d_model: int = 256):
        super().__init__()
        self.encoder = encoder
        self.textdec = textdec
        self.img_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 512),
        )

    def forward(self, batch, enc_fn, clip_image_emb_fn):
        ctx_images = batch["ctx_images"]
        tgt_image = batch["tgt_image"]
        z = self.encoder(ctx_images, batch["ctx_captions"])
        enc, _ = enc_fn(batch["ctx_captions"], batch["tgt_caption"])
        input_ids = enc["input_ids"].to(ctx_images.device)
        attn_mask = enc["attention_mask"].to(ctx_images.device)
        labels = enc["labels"].to(ctx_images.device)

        text_loss, txt_emb, _ = self.textdec(input_ids, attn_mask, labels, z)
        with torch.no_grad():
            tgt_img_emb = clip_image_emb_fn(tgt_image)
        pred_img_emb = F.normalize(self.img_head(z), dim=-1)
        img_loss = F.mse_loss(pred_img_emb, tgt_img_emb)
        align_loss = (1.0 - (pred_img_emb * txt_emb).sum(dim=-1)).mean()
        return text_loss, img_loss, align_loss
