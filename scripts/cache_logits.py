"""Run both experts once over the evaluation data and cache their logits.

    python scripts/cache_logits.py --cifar data/cifar-10-batches-py --imdb data/imdb_test.csv

Everything downstream (calibration, thresholds, evaluation, figures) works
from these cached logits, so it is fast to iterate on.

Data
----
* CIFAR-10 test batch (10,000 images), python version from
  https://www.cs.toronto.edu/~kriz/cifar.html
* IMDb test split as CSV with ``text,label`` columns (``datasets.load_dataset("stanfordnlp/imdb")``)
* Out-of-distribution images are generated offline (no download needed):
  handwritten digits, crops of natural photographs, and noise.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metacogai.experts import ImageExpert, TextExpert  # noqa: E402


def load_cifar_test(folder: Path):
    with open(folder / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = torch.tensor(d[b"data"], dtype=torch.float32).view(-1, 3, 32, 32) / 255.0
    return (x - 0.5) / 0.5, np.array(d[b"labels"])


def _to_tensor(arr_uint8_hw3):
    img = Image.fromarray(arr_uint8_hw3)
    return ImageExpert.preprocess(img)


def ood_images(n_each: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Images that do not belong to any CIFAR-10 class."""
    from sklearn.datasets import load_digits, load_sample_images

    # 1) handwritten digits, upscaled and given random ink/paper colours
    digits = load_digits().images                                   # 8x8, 0..16
    idx = rng.choice(len(digits), n_each, replace=False)
    out_digits = []
    for i in idx:
        g = (digits[i] / 16.0)[..., None]
        ink, paper = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        rgb = (g * ink + (1 - g) * paper).astype(np.uint8)
        out_digits.append(_to_tensor(np.array(Image.fromarray(rgb).resize((32, 32), Image.Resampling.NEAREST))))

    # 2) crops of natural photographs (a Chinese temple, a flower) -- textures and scenes
    photos = load_sample_images().images
    out_crops = []
    for _ in range(n_each):
        p = photos[rng.integers(len(photos))]
        s = int(rng.integers(48, 200))
        y0, x0 = rng.integers(0, p.shape[0] - s), rng.integers(0, p.shape[1] - s)
        out_crops.append(_to_tensor(p[y0:y0 + s, x0:x0 + s]))

    # 3) noise
    noise = torch.clamp(torch.randn(n_each, 3, 32, 32) * 0.5, -1, 1)
    return {"digits": torch.stack(out_digits), "photo_crops": torch.stack(out_crops), "noise": noise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar", type=Path, required=True)
    ap.add_argument("--imdb", type=Path, required=True)
    ap.add_argument("--n-text", type=int, default=4000, help="IMDb reviews to score (DistilBERT is slow on CPU)")
    ap.add_argument("--n-ood", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/cache"))
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    img = ImageExpert()
    x, y = load_cifar_test(args.cifar)
    f, z = img.features_and_logits(x)
    np.savez_compressed(args.out / "cifar_test.npz", logits=z.numpy(), labels=y)
    np.savez_compressed(args.out / "cifar_test_features.npz",
                        **{f"layer{i + 1}": v.numpy().astype(np.float16) for i, v in enumerate(f)})
    ood = ood_images(args.n_ood, rng)
    ood_out = {k: img.features_and_logits(v) for k, v in ood.items()}
    np.savez_compressed(args.out / "ood_images.npz", **{k: v[1].numpy() for k, v in ood_out.items()})
    np.savez_compressed(args.out / "ood_features.npz",
                        **{f"{k}__layer{i + 1}": f.numpy().astype(np.float16)
                           for k, v in ood_out.items() for i, f in enumerate(v[0])})
    print("image logits cached")

    if not args.skip_text:
        df = pd.read_csv(args.imdb).sample(args.n_text, random_state=0)
        txt = TextExpert()
        z = txt.logits(list(df.text))
        np.savez_compressed(args.out / "imdb_test.npz", logits=z.numpy(), labels=df.label.values,
                            index=df.index.values)
        print("text logits cached")


if __name__ == "__main__":
    main()
