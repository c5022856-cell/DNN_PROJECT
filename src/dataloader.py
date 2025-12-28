"""
Dataset utilities adapted from the notebook.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

_GDI = re.compile(r"<gdi\\s+image(\\d+)>", re.IGNORECASE)
_GD_TAGS = re.compile(r"</?gd[ioal]\\b[^>]*>", re.IGNORECASE)
_ANY_TAGS = re.compile(r"<[^>]+>")


def strip_grounding_tags(text: str) -> str:
    if text is None:
        return ""
    text = _GD_TAGS.sub("", text)
    text = _ANY_TAGS.sub("", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def split_story_into_frame_segments(story_text: str) -> Dict[int, str]:
    if not story_text:
        return {}
    matches = list(_GDI.finditer(story_text))
    if not matches:
        return {}
    segs: Dict[int, str] = {}
    for i, m in enumerate(matches):
        frame_idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(story_text)
        segs[frame_idx] = story_text[start:end].strip()
    return segs


def get_frame_captions(example: Dict[str, Any]) -> List[str]:
    frame_count = int(example.get("frame_count", 0))
    segs = split_story_into_frame_segments(example.get("story", ""))
    captions = []
    for i in range(1, frame_count + 1):
        captions.append(strip_grounding_tags(segs.get(i, "")))
    return captions


def build_stratified_split_indices(hf_ds, K: int = 4, seed: int = 42):
    def has_k_plus_1(ex):
        return int(ex["frame_count"]) >= (K + 1)

    train_ok = hf_ds.filter(has_k_plus_1)
    df_ids = pd.DataFrame(
        {
            "idx": list(range(len(train_ok))),
            "story_id": [train_ok[i]["story_id"] for i in range(len(train_ok))],
            "frame_count": [int(train_ok[i]["frame_count"]) for i in range(len(train_ok))],
        }
    )
    df_ids["len_bin"] = pd.cut(df_ids["frame_count"], bins=[0, 7, 11, 15, 100], labels=["5-7", "8-11", "12-15", "16+"])
    df_ids = df_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for _, group in df_ids.groupby("len_bin", observed=False):
        idxs = group["idx"].tolist()
        n = len(idxs)
        n_train = int(round(0.80 * n))
        n_val = int(round(0.10 * n))
        train_idx += idxs[:n_train]
        val_idx += idxs[n_train : n_train + n_val]
        test_idx += idxs[n_train + n_val :]

    return train_ok, train_idx, val_idx, test_idx


def build_image_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )


class StoryKPlus1(Dataset):
    def __init__(self, hf_ds, indices: Sequence[int], K: int = 4, img_tf=None):
        self.hf_ds = hf_ds
        self.indices = list(indices)
        self.K = K
        self.img_tf = img_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        ex = self.hf_ds[self.indices[i]]
        captions = get_frame_captions(ex)
        images = ex["images"]

        ctx_imgs = images[: self.K]
        tgt_img = images[self.K]

        if self.img_tf is not None:
            ctx_imgs = torch.stack([self.img_tf(im.convert("RGB")) for im in ctx_imgs], dim=0)
            tgt_img = self.img_tf(tgt_img.convert("RGB"))

        return {
            "story_id": ex["story_id"],
            "frame_count": int(ex["frame_count"]),
            "ctx_images": ctx_imgs,
            "ctx_captions": captions[: self.K],
            "tgt_image": tgt_img,
            "tgt_caption": captions[self.K],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctx_images = torch.stack([b["ctx_images"] for b in batch], dim=0)
    tgt_image = torch.stack([b["tgt_image"] for b in batch], dim=0)
    return {
        "story_id": [b["story_id"] for b in batch],
        "frame_count": torch.tensor([b["frame_count"] for b in batch]),
        "ctx_images": ctx_images,
        "tgt_image": tgt_image,
        "ctx_captions": [b["ctx_captions"] for b in batch],
        "tgt_caption": [b["tgt_caption"] for b in batch],
    }
