# Multimodal Alignment for Visual Story Continuation

Purpose
- Align predicted captions with target images using shared CLIP-based embeddings and a prefix-conditioned GPT-2 decoder.

Method overview
- Dataset: StoryReasoning, with per-frame captions parsed from `<gdi imageX>` segments.
- Encoder: CLIP image + CLIP text embeddings fused and temporally aggregated via a Transformer encoder.
- Decoder: GPT-2 with prefix conditioning from the shared context vector.
- Losses: text loss, image embedding regression, and cosine alignment between predicted image and text embeddings.

Repository layout
- `src/dataloader.py`: caption parsing, stratified split, dataset and collation.
- `src/encoders_image.py`: CLIP image embedding helper.
- `src/encoders_text.py`: GPT-2 tokenization and CLIP text embeddings.
- `src/generator.py`: shared encoder, prefix GPT-2, and alignment model.
- `src/discriminator.py`: alignment loss helper.
- `src/train_gan.py`: training loop for alignment model.
- `src/eval_gan.py`: evaluation helpers.
- `src/datadownload.py`: dataset download/inspection.

Quickstart
```bash
cd "Ashraf - Project 5"

# Train alignment model
python src/train_gan.py --epochs 2 --batch_size 4
```

Notes
- Captions are derived from `<gdi imageX>` segments with grounding tags stripped.
- CLIP is frozen; only lightweight fusion, prefix, and heads are trained.
