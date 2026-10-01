"""Metacognition: each expert's judgement about its *own* prediction.

For every input an expert answers four questions:

1. **What do I think?** -- the predicted label.
2. **How sure should I be?** -- a *calibrated* probability. Raw softmax scores
   are over-confident (the text model averages 98% confidence at 92%
   accuracy), so a temperature is fitted on held-out data (Guo et al., 2017).
3. **Is this even my kind of input?** -- out-of-distribution (OOD) checks.
   For images: the energy score of the logits (Liu et al., 2020) plus the
   Mahalanobis distance of the features at *each* of the four ResNet stages
   to the training classes (Lee et al., 2018). Early layers catch inputs with
   the wrong low-level statistics (noise, line drawings); the energy score
   catches inputs the classifier finds unfamiliar as a whole. Each score is
   standardised on in-distribution calibration images and the image is
   flagged if the *least* typical one falls below a threshold that only 5%
   of real CIFAR-10 images cross.
4. **Should I answer or abstain?** -- a confidence threshold chosen on held-out
   data so that answered predictions reach a target accuracy (selective
   prediction; Geifman & El-Yaniv, 2017).

All thresholds live in ``weights/metacog_calibration.json`` and are produced by
``scripts/calibrate.py`` -- nothing is hand-tuned.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .experts import WEIGHTS

CALIBRATION_FILE = WEIGHTS / "metacog_calibration.json"
MAHALANOBIS_FILE = WEIGHTS / "image_feature_stats.npz"


# ----------------------------------------------------------------- maths ----
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    """Normalised predictive entropy in [0, 1] (1 = maximally unsure)."""
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(-1) / np.log(p.shape[-1])


def energy(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Negative free energy, T * logsumexp(z / T). Higher = more in-distribution."""
    z = np.asarray(z, dtype=np.float64)
    m = z.max(-1, keepdims=True)
    return (T * (np.log(np.exp((z - m) / T).sum(-1)) + m[..., 0] / T))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def selective_threshold(conf: np.ndarray, correct: np.ndarray, target_acc: float) -> float:
    """Lowest confidence threshold whose answered set reaches ``target_acc``."""
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(cum_acc >= target_acc)
    if not len(ok):
        return 1.0
    k = ok.max()                       # largest answered set that still meets the target
    return float(conf[order][k])


class Mahalanobis:
    """Class-conditional Gaussian with shared covariance on one layer's features."""

    def __init__(self, means: np.ndarray, precision: np.ndarray):
        self.means, self.precision = means, precision

    @classmethod
    def fit(cls, feats: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3):
        k = labels.max() + 1
        means = np.stack([feats[labels == c].mean(0) for c in range(k)])
        centred = feats - means[labels]
        cov = centred.T @ centred / len(centred) + shrinkage * np.eye(feats.shape[1])
        return cls(means.astype(np.float32), np.linalg.inv(cov).astype(np.float32))

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Negative squared distance to the closest class mean (higher = more typical)."""
        d = feats[:, None, :] - self.means[None]
        return -np.einsum("nkd,de,nke->nk", d, self.precision, d).min(1)


class FamiliarityScorer:
    """Combines per-layer Mahalanobis scores and the energy score into one OOD score."""

    def __init__(self, layers: list[Mahalanobis], mean: np.ndarray, std: np.ndarray):
        self.layers, self.mean, self.std = layers, mean, std
        self.names = [f"layer{i + 1}" for i in range(len(layers))] + ["energy"]

    def raw_scores(self, feats: list[np.ndarray], logits: np.ndarray) -> np.ndarray:
        """[N, n_layers + 1] -- higher = more familiar."""
        return np.stack([m.score(f) for m, f in zip(self.layers, feats)] + [energy(logits)], 1)

    def standardised(self, feats, logits) -> np.ndarray:
        return (self.raw_scores(feats, logits) - self.mean) / self.std

    def score(self, feats, logits) -> np.ndarray:
        """The least typical standardised score (a z-score; very negative = unfamiliar)."""
        return self.standardised(feats, logits).min(1)

    def save(self, path: Path = MAHALANOBIS_FILE):
        arrays = {"mean": self.mean, "std": self.std}
        for i, m in enumerate(self.layers):
            arrays[f"means_{i}"], arrays[f"precision_{i}"] = m.means, m.precision
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path = MAHALANOBIS_FILE):
        d = np.load(path)
        n = sum(k.startswith("means_") for k in d.files)
        return cls([Mahalanobis(d[f"means_{i}"], d[f"precision_{i}"]) for i in range(n)], d["mean"], d["std"])


# ------------------------------------------------------------ assessment ----
@dataclass
class SelfAssessment:
    modality: str
    label: str
    raw_confidence: float          # plain softmax
    confidence: float              # temperature-calibrated
    uncertainty: float             # normalised entropy of calibrated probs
    in_domain: bool
    answer: bool                   # False = the expert abstains
    reasons: list[str] = field(default_factory=list)
    top: list[tuple[str, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Metacognition:
    """Holds the calibration state and turns logits into ``SelfAssessment``s."""

    def __init__(self, cfg: dict, familiarity: FamiliarityScorer | None = None):
        self.cfg, self.familiarity = cfg, familiarity

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE):
        cfg = json.loads(Path(path).read_text())
        fam = FamiliarityScorer.load() if MAHALANOBIS_FILE.exists() else None
        return cls(cfg, fam)

    # ---------------------------------------------------------------- image
    def assess_image(self, logits: np.ndarray, features: list[np.ndarray] | None, labels) -> SelfAssessment:
        c = self.cfg["image"]
        p_raw, p = softmax(logits), softmax(logits, c["temperature"])
        k = int(p.argmax())
        reasons, extra = [], {}
        in_domain = True
        if self.familiarity is not None and features is not None:
            z = self.familiarity.standardised([f[None] for f in features], logits[None])[0]
            worst = int(z.argmin())
            extra = {"familiarity": float(z.min()), "least_familiar_signal": self.familiarity.names[worst],
                     "familiarity_by_signal": dict(zip(self.familiarity.names, map(float, z)))}
            in_domain = bool(z.min() >= c["familiarity_threshold"])
            if not in_domain:
                where = ("its low-level texture and colour statistics" if worst < 2 else
                         "its high-level features" if worst < 4 else "the classifier's overall response")
                reasons.append(f"this image is unlike anything it was trained on ({where} are atypical "
                               f"for all 10 classes)")
        conf_ok = bool(p[k] >= c["answer_threshold"])
        if not conf_ok:
            reasons.append(f"calibrated confidence {p[k]:.0%} is below the {c['answer_threshold']:.0%} needed "
                           f"to be right {c['target_accuracy']:.0%} of the time")
        top = sorted(zip(labels, p.tolist()), key=lambda t: -t[1])[:3]
        return SelfAssessment("image", labels[k], float(p_raw[k]), float(p[k]), float(entropy(p)),
                              in_domain, in_domain and conf_ok, reasons, top, extra)

    # ----------------------------------------------------------------- text
    def assess_text(self, logits: np.ndarray, labels, mc_probs: np.ndarray | None = None) -> SelfAssessment:
        c = self.cfg["text"]
        p_raw, p = softmax(logits), softmax(logits, c["temperature"])
        k = int(p.argmax())
        reasons, extra = [], {}
        conf_ok = bool(p[k] >= c["answer_threshold"])
        if not conf_ok:
            reasons.append(f"calibrated confidence {p[k]:.0%} is below the {c['answer_threshold']:.0%} needed "
                           f"to be right {c['target_accuracy']:.0%} of the time")
        if mc_probs is not None:
            # how often does the prediction flip when parts of the network are switched off?
            flips = float((mc_probs.argmax(1) != k).mean())
            extra["mc_dropout_disagreement"] = flips
            if flips >= 0.2:
                reasons.append(f"the prediction flips in {flips:.0%} of dropout samples, so the review is probably mixed")
                conf_ok = False
        top = sorted(zip(labels, p.tolist()), key=lambda t: -t[1])
        return SelfAssessment("text", labels[k], float(p_raw[k]), float(p[k]), float(entropy(p)),
                              True, conf_ok, reasons, top, extra)
