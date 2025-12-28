"""
CLIP image embedding helpers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class ClipImageEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_tensor_01(self, img_bchw_01: torch.Tensor) -> torch.Tensor:
        """
        img_bchw_01: tensor in [0,1], shape (B,3,H,W)
        """
        # Use CLIPProcessor to normalize as expected
        imgs = []
        for i in range(img_bchw_01.size(0)):
            arr = (img_bchw_01[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            imgs.append(arr)
        inputs = self.processor(images=imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        emb = self.clip.get_image_features(pixel_values=pixel_values)
        return F.normalize(emb, dim=-1)
