"""
CLIP text embedding and GPT-2 tokenization helpers.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor


def load_gpt2(name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(name)
    return tok, gpt


def load_clip_text(model_name: str = "openai/clip-vit-base-patch32", device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    for p in clip.parameters():
        p.requires_grad = False
    return clip, processor, device


@torch.no_grad()
def clip_text_emb(clip: CLIPModel, processor: CLIPProcessor, device: torch.device, text_list: List[str]) -> torch.Tensor:
    inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return F.normalize(emb, dim=-1)


def format_prompt(ctx_captions: List[str], max_ctx_words: int = 25) -> str:
    def compress(c: str) -> str:
        if not c:
            return ""
        w = c.split()
        return " ".join(w[:max_ctx_words])

    lines = [f"{i+1}: {compress(c)}" for i, c in enumerate(ctx_captions)]
    return "CTX\n" + "\n".join(lines) + "\nNEXT:"


def tokenize_batch(
    tok,
    ctx_captions_batch: List[List[str]],
    tgt_caption_batch: List[str],
    max_len: int = 256,
    max_tgt_tok: int = 64,
    max_ctx_words: int = 25,
) -> Tuple[dict, List[str]]:
    prompts = [format_prompt(caps, max_ctx_words=max_ctx_words) for caps in ctx_captions_batch]
    targets = [t if t else "" for t in tgt_caption_batch]

    max_prompt_len = max_len - max_tgt_tok
    prompt_enc = tok(prompts, padding=False, truncation=True, max_length=max_prompt_len, add_special_tokens=False)
    tgt_enc = tok(targets, padding=False, truncation=True, max_length=max_tgt_tok, add_special_tokens=False)

    input_ids_list = []
    labels_list = []
    for p_ids, t_ids in zip(prompt_enc["input_ids"], tgt_enc["input_ids"]):
        if len(t_ids) == 0:
            t_ids = [tok.eos_token_id]
        ids = (p_ids + t_ids)[:max_len]
        labels = ([-100] * len(p_ids) + t_ids)[:max_len]
        input_ids_list.append(torch.tensor(ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    max_b = max(x.size(0) for x in input_ids_list)
    input_ids = torch.full((len(input_ids_list), max_b), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_b), -100, dtype=torch.long)
    attn_mask = torch.zeros((len(input_ids_list), max_b), dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        n = ids.size(0)
        input_ids[i, :n] = ids
        labels[i, :n] = lab
        attn_mask[i, :n] = 1

    supervised = (labels != -100).sum(dim=1)
    if (supervised == 0).any():
        bad = (supervised == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"Found samples with 0 supervised tokens: {bad}")

    enc = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
    return enc, prompts
