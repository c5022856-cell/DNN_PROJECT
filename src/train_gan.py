"""
Train alignment model (shared encoder + GPT-2 prefix + CLIP alignment).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

from .dataloader import StoryKPlus1, build_image_transform, build_stratified_split_indices, collate_fn
from .encoders_image import ClipImageEncoder
from .encoders_text import load_clip_text, load_gpt2, tokenize_batch
from .generator import SharedEncoder, PrefixGPT2, AlignmentModel


def main():
    parser = argparse.ArgumentParser(description="Train alignment model.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha_img", type=float, default=0.5)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="checkpoints/alignment_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("daniel3303/StoryReasoning", cache_dir=args.cache_dir)

    train_ok, train_idx, val_idx, test_idx = build_stratified_split_indices(ds["train"], K=args.k_steps)
    img_tf = build_image_transform(224)

    ds_train = StoryKPlus1(train_ok, train_idx, K=args.k_steps, img_tf=img_tf)
    ds_val = StoryKPlus1(train_ok, val_idx, K=args.k_steps, img_tf=img_tf)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    clip_image = ClipImageEncoder(device=device)
    clip, clip_proc, device = load_clip_text(device=device)
    tok, gpt = load_gpt2()
    gpt = gpt.to(device)

    def clip_text_emb(texts):
        inputs = clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return F.normalize(emb, dim=-1)

    encoder = SharedEncoder(clip_image.embed_tensor_01, clip_text_emb, d_model=256)
    encoder = encoder.to(device)
    textdec = PrefixGPT2(gpt, d_z=256, n_prefix=8).to(device)
    model = AlignmentModel(encoder, textdec, d_model=256).to(device)

    for p in model.textdec.gpt.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    def run_epoch(dl, train=True):
        model.train(train)
        total = {"txt": 0.0, "img": 0.0, "align": 0.0, "tot": 0.0}
        n = 0
        pbar = tqdm(dl, desc=("train" if train else "val"))
        for batch in pbar:
            batch["ctx_images"] = batch["ctx_images"].to(device)
            batch["tgt_image"] = batch["tgt_image"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                txt_loss, img_loss, align_loss = model(batch, lambda c, t: tokenize_batch(tok, c, t), clip_image.embed_tensor_01)
                loss = txt_loss + args.alpha_img * img_loss + args.lambda_align * align_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            bs = batch["tgt_image"].shape[0]
            total["txt"] += txt_loss.detach().item() * bs
            total["img"] += img_loss.detach().item() * bs
            total["align"] += align_loss.detach().item() * bs
            total["tot"] += loss.detach().item() * bs
            n += bs
            pbar.set_postfix(tot=total["tot"] / n, txt=total["txt"] / n, img=total["img"] / n, al=total["align"] / n)
        for k in total:
            total[k] /= max(n, 1)
        return total

    best_val = 1e9
    save_path = Path(args.save_path)
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(dl_train, train=True)
        va = run_epoch(dl_val, train=False)
        print(f"Epoch {ep} | train:", tr, "| val:", va)
        if va["tot"] < best_val:
            best_val = va["tot"]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": ep, "model_state": model.state_dict(), "opt_state": optimizer.state_dict()}, str(save_path))
            print("  saved", str(save_path))


if __name__ == "__main__":
    main()
