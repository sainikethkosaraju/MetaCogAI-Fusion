"""Fit the metacognition layer on held-out calibration data.

    python scripts/calibrate.py --cifar data/cifar-10-batches-py

Reads cached logits (``scripts/cache_logits.py``), uses the *calibration half*
of each test set, and writes:

* ``weights/metacog_calibration.json`` -- temperatures and decision thresholds
* ``weights/image_feature_stats.npz``  -- class means / precision for Mahalanobis OOD

The other half of each test set is never touched here; ``scripts/evaluate.py``
reports results on it.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metacogai.experts import ImageExpert  # noqa: E402
from metacogai.metacognition import (CALIBRATION_FILE, FamiliarityScorer, Mahalanobis,  # noqa: E402
                                     fit_temperature, selective_threshold, softmax)

SPLIT_SEED = 0
TARGET_ACCURACY = 0.95       # answered predictions should be right at least this often
OOD_FALSE_ALARM = 0.05       # share of real CIFAR-10 calibration images flagged as unfamiliar


def split(n: int, seed: int = SPLIT_SEED):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: n // 2], perm[n // 2:]


def load_cifar_train(folder: Path):
    xs, ys = [], []
    for b in range(1, 6):
        with open(folder / f"data_batch_{b}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(torch.tensor(d[b"data"], dtype=torch.float32).view(-1, 3, 32, 32) / 255.0)
        ys += d[b"labels"]
    return (torch.cat(xs) - 0.5) / 0.5, np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar", type=Path, required=True, help="folder with CIFAR-10 python batches")
    ap.add_argument("--cache", type=Path, default=ROOT / "results/cache")
    args = ap.parse_args()

    # ---------------- image
    c = np.load(args.cache / "cifar_test.npz")
    feats = np.load(args.cache / "cifar_test_features.npz")
    cal, _ = split(len(c["labels"]))
    z, y = c["logits"][cal], c["labels"][cal]
    f = [feats[f"layer{i}"][cal].astype(np.float32) for i in range(1, 5)]
    T_img = fit_temperature(z, y)
    p = softmax(z, T_img)
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)

    print("fitting Mahalanobis statistics on the 50,000 CIFAR-10 training images ...")
    x_tr, y_tr = load_cifar_train(args.cifar)
    f_tr, _ = ImageExpert().features_and_logits(x_tr)
    layers = [Mahalanobis.fit(ft.numpy(), y_tr) for ft in f_tr]
    raw = FamiliarityScorer(layers, 0, 1).raw_scores(f, z)
    fam = FamiliarityScorer(layers, raw.mean(0), raw.std(0))
    fam.save()

    image_cfg = {
        "temperature": T_img,
        "target_accuracy": TARGET_ACCURACY,
        "answer_threshold": selective_threshold(conf, correct, TARGET_ACCURACY),
        "familiarity_threshold": float(np.quantile(fam.score(f, z), OOD_FALSE_ALARM)),
    }

    # ---------------- text
    t = np.load(args.cache / "imdb_test.npz")
    cal_t, _ = split(len(t["labels"]))
    zt, yt = t["logits"][cal_t], t["labels"][cal_t]
    T_txt = fit_temperature(zt, yt)
    pt = softmax(zt, T_txt)
    text_cfg = {
        "temperature": T_txt,
        "target_accuracy": TARGET_ACCURACY,
        "answer_threshold": selective_threshold(pt.max(1), (pt.argmax(1) == yt).astype(float), TARGET_ACCURACY),
    }

    cfg = {"image": image_cfg, "text": text_cfg,
           "fusion": {"mention_reliability": 0.8},
           "_provenance": {"split_seed": SPLIT_SEED, "n_image_calibration": int(len(cal)),
                           "n_text_calibration": int(len(cal_t)),
                           "ood_false_alarm_rate": OOD_FALSE_ALARM}}
    CALIBRATION_FILE.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
