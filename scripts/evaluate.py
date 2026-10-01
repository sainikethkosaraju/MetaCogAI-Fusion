"""Evaluate the metacognitive layer on the held-out evaluation half.

    python scripts/evaluate.py

Writes ``results/metrics.json`` and figures in ``results/figures/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metacogai.experts import CIFAR10_CLASSES  # noqa: E402
from metacogai.fusion import legacy_fuse, text_evidence_posterior  # noqa: E402
from metacogai.metacognition import CALIBRATION_FILE, FamiliarityScorer, ece, softmax  # noqa: E402
from scripts.calibrate import split  # noqa: E402

CACHE, OUT = ROOT / "results/cache", ROOT / "results"
FIG = OUT / "figures"
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9.5,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe8", "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "lines.linewidth": 2,
})


def risk_coverage(conf, correct):
    order = np.argsort(-conf)
    c = correct[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, acc


def aurc(conf, correct):
    cov, acc = risk_coverage(conf, correct)
    return float(np.mean(1 - acc))


def calib_block(z, y, T, threshold):
    out = {}
    for name, t in [("raw", 1.0), ("calibrated", T)]:
        p = softmax(z, t)
        conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        out[name] = {"accuracy": float(corr.mean()), "mean_confidence": float(conf.mean()),
                     "ece": ece(conf, corr), "nll": nll}
    p = softmax(z, T)
    conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
    answered = conf >= threshold
    out["selective"] = {"threshold": threshold, "coverage": float(answered.mean()),
                        "accuracy_when_answering": float(corr[answered].mean()),
                        "accuracy_of_abstained": float(corr[~answered].mean()) if (~answered).any() else None,
                        "aurc": aurc(conf, corr)}
    return out, conf, corr


def reliability_plot(ax, z, y, T, title):
    for t, color, lab in [(1.0, GREY, "raw softmax"), (T, BLUE, f"calibrated (T={T:.2f})")]:
        p = softmax(z, t)
        conf, corr = p.max(1), (p.argmax(1) == y)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ys = [corr[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ax.plot(xs, ys, marker="o", ms=5, color=color, label=f"{lab}, ECE {ece(conf, corr.astype(float)):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, loc="upper left")


def main():
    cfg = json.loads(CALIBRATION_FILE.read_text())
    ci, ct = cfg["image"], cfg["text"]
    metrics = {}

    # ---------------------------------------------------------------- image
    c = np.load(CACHE / "cifar_test.npz")
    feats = np.load(CACHE / "cifar_test_features.npz")
    _, ev = split(len(c["labels"]))
    zi, yi = c["logits"][ev], c["labels"][ev]
    fi = [feats[f"layer{i}"][ev].astype(np.float32) for i in range(1, 5)]
    metrics["image"], conf_i, corr_i = calib_block(zi, yi, ci["temperature"], ci["answer_threshold"])

    # ---------------------------------------------------------------- text
    t = np.load(CACHE / "imdb_test.npz")
    _, ev_t = split(len(t["labels"]))
    zt, yt = t["logits"][ev_t], t["labels"][ev_t]
    metrics["text"], conf_t, corr_t = calib_block(zt, yt, ct["temperature"], ct["answer_threshold"])

    # ---------------------------------------------------------------- OOD
    fam = FamiliarityScorer.load()
    ood_z, ood_f = np.load(CACHE / "ood_images.npz"), np.load(CACHE / "ood_features.npz")
    thr = ci["familiarity_threshold"]
    std_in = fam.standardised(fi, zi)
    s_in = std_in.min(1)
    metrics["ood"] = {"false_alarm_rate_on_cifar": float((s_in < thr).mean()), "sets": {}}
    ood_scores = {}
    for name in ood_z.files:
        zo = ood_z[name]
        fo = [ood_f[f"{name}__layer{i}"].astype(np.float32) for i in range(1, 5)]
        std_o = fam.standardised(fo, zo)
        s_o = std_o.min(1)
        ood_scores[name] = s_o
        lab = np.r_[np.ones(len(s_in)), np.zeros(len(s_o))]
        raw_conf = softmax(zo).max(1)
        per_signal = {n: float(roc_auc_score(lab, np.r_[std_in[:, j], std_o[:, j]]))
                      for j, n in enumerate(fam.names)}
        metrics["ood"]["sets"][name] = {
            "auroc_combined": float(roc_auc_score(lab, np.r_[s_in, s_o])),
            "auroc_max_softmax": float(roc_auc_score(lab, np.r_[softmax(zi).max(1), raw_conf])),
            "auroc_by_signal": per_signal,
            "detected": float((s_o < thr).mean()),
            "raw_model_mean_confidence": float(raw_conf.mean()),
            "raw_model_answers_over_90pct": float((raw_conf >= 0.9).mean()),
            "answered_after_metacognition": float(((s_o >= thr) & (softmax(zo, ci["temperature"]).max(1)
                                                                   >= ci["answer_threshold"])).mean()),
        }

    # ------------------------------------------------- cross-modal evidence
    rng = np.random.default_rng(1)
    p_img = softmax(zi, ci["temperature"])
    r = cfg["fusion"]["mention_reliability"]
    xm = {}
    for true_rel in (0.95, 0.8, 0.6):
        names_true = rng.random(len(yi)) < true_rel
        other = (yi + rng.integers(1, 10, len(yi))) % 10
        mention = np.where(names_true, yi, other)
        post = np.stack([text_evidence_posterior(p_img[i], [CIFAR10_CLASSES[mention[i]]], r)
                         for i in range(len(yi))])
        xm[str(true_rel)] = {"image_only_accuracy": float((p_img.argmax(1) == yi).mean()),
                             "with_text_accuracy": float((post.argmax(1) == yi).mean())}
    metrics["cross_modal_simulation"] = {"assumed_reliability": r, "results_by_true_caption_reliability": xm}

    # ----------------------------------- legacy fusion rule vs metacognition
    n = min(len(yi), len(yt))
    pi_raw, pt_raw = softmax(zi[:n]).max(1), softmax(zt[:n]).max(1)
    leg = [legacy_fuse((CIFAR10_CLASSES[a], b), (("positive" if c_ else "negative"), d))
           for a, b, c_, d in zip(zi[:n].argmax(1), pi_raw, zt[:n].argmax(1), pt_raw)]
    picked_text = np.array([lab in ("positive", "negative") for lab, _ in leg])
    deferred = np.array([lab == "Uncertain" for lab, _ in leg])
    ans_i, ans_t = conf_i[:n] >= ci["answer_threshold"], conf_t[:n] >= ct["answer_threshold"]
    metrics["fusion_pairs"] = {
        "n_pairs": int(n),
        "legacy_rule": {"deferred": float(deferred.mean()), "reported_only_text": float(picked_text.mean()),
                        "reported_only_image": float((~picked_text & ~deferred).mean()),
                        "note": "the legacy rule always discards one modality"},
        "metacognitive": {"both_answered": float((ans_i & ans_t).mean()),
                          "accuracy_when_both_answered": float((corr_i[:n] * corr_t[:n])[ans_i & ans_t].mean()),
                          "partial": float((ans_i ^ ans_t).mean()), "deferred": float((~ans_i & ~ans_t).mean()),
                          "raw_joint_accuracy_if_always_answering": float((corr_i[:n] * corr_t[:n]).mean())},
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------------------------------------------------------- figures
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    reliability_plot(axes[0], zi, yi, ci["temperature"], "Image expert (CIFAR-10)")
    reliability_plot(axes[1], zt, yt, ct["temperature"], "Text expert (IMDb)")
    fig.suptitle("Does stated confidence match actual accuracy?", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, conf, corr, a_thr, title in [(axes[0], conf_i, corr_i, ci["answer_threshold"], "Image expert"),
                                         (axes[1], conf_t, corr_t, ct["answer_threshold"], "Text expert")]:
        cov, acc = risk_coverage(conf, corr)
        ax.plot(cov * 100, acc * 100, color=BLUE)
        k = (conf >= a_thr).sum()
        ax.scatter([k / len(conf) * 100], [corr[conf >= a_thr].mean() * 100], color=ORANGE, zorder=3, s=40)
        ax.annotate(f"operating point\nanswers {k / len(conf):.0%}, right {corr[conf >= a_thr].mean():.1%}",
                    (k / len(conf) * 100, corr[conf >= a_thr].mean() * 100), xytext=(-110, -38),
                    textcoords="offset points", fontsize=8, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        ax.axhline(corr.mean() * 100, color=GREY, ls="--", lw=1)
        ax.text(2, corr.mean() * 100, f"always answering: {corr.mean():.1%}", fontsize=8, color=MUTED, va="bottom")
        ax.set_xlabel("coverage: % of inputs answered")
        ax.set_ylabel("accuracy on answered inputs (%)")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 101)
    fig.suptitle("Knowing when to abstain: accuracy rises as the least-confident inputs are declined",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    colors = {"digits": ORANGE, "photo_crops": AQUA, "noise": YELLOW}
    lo = np.quantile(np.concatenate([s_in, *ood_scores.values()]), 0.01)
    bins = np.linspace(lo, np.quantile(s_in, 0.999), 60)
    ax.hist(np.clip(s_in, lo, None), bins=bins, color=BLUE, alpha=0.55, density=True, label="CIFAR-10 test (in-domain)")
    for name, v in ood_scores.items():
        ax.hist(np.clip(v, lo, None), bins=bins, histtype="step", lw=1.6, color=colors[name], density=True,
                label=f"{name.replace('_', ' ')}: {(v < thr).mean():.0%} flagged")
    ax.axvline(thr, color=INK, lw=1, ls="--")
    ax.text(thr, ax.get_ylim()[1] * 0.95, "  threshold (5% false alarms)", fontsize=8, color=INK, va="top")
    ax.set_yticks([])
    ax.set_xlabel("familiarity score (least typical of 4 feature layers + energy; z-score)")
    ax.set_title("Recognising images unlike anything it was trained on", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "ood_scores.png")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
