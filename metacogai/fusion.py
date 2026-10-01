"""Metacognitive fusion of an image and a piece of text (e.g. a photo and its review).

The two experts answer different questions -- *what is in the picture?* and
*how does the writer feel?* -- so fusion does three things:

1. **Cross-modal evidence.** If the text names one of the image classes
   ("my puppy", "this old truck"), that mention is treated as a noisy
   observation of the true class and combined with the image posterior by
   Bayes' rule. The text can therefore *correct* the image model when the
   picture is ambiguous, and the report says so.
2. **Conflict detection.** If the text names a class the image model rules
   out, the system reports a cross-modal conflict instead of silently
   picking a side.
3. **Joint self-assessment.** The joint confidence is the product of the two
   calibrated confidences (the experts' errors are independent), and the
   system answers fully, partially (only the trustworthy half), or defers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .experts import CIFAR10_CLASSES
from .metacognition import SelfAssessment

# words that refer to each CIFAR-10 class
CLASS_WORDS = {
    "airplane": ["airplane", "aeroplane", "plane", "jet", "aircraft", "airliner"],
    "automobile": ["automobile", "car", "sedan", "suv", "hatchback", "coupe"],
    "bird": ["bird", "parrot", "sparrow", "pigeon", "eagle", "owl", "ostrich", "finch"],
    "cat": ["cat", "kitten", "kitty", "tabby"],
    "deer": ["deer", "fawn", "stag", "doe", "elk", "moose", "reindeer"],
    "dog": ["dog", "puppy", "pup", "doggo", "retriever", "terrier", "labrador", "poodle"],
    "frog": ["frog", "toad", "tadpole"],
    "horse": ["horse", "pony", "stallion", "mare", "foal"],
    "ship": ["ship", "boat", "vessel", "ferry", "yacht", "cruise", "tanker"],
    "truck": ["truck", "lorry", "pickup", "dump truck", "fire truck", "18-wheeler"],
}
_PATTERNS = {c: re.compile(r"\b(" + "|".join(re.escape(w) for w in ws) + r")s?\b", re.I)
             for c, ws in CLASS_WORDS.items()}


def mentioned_classes(text: str) -> list[str]:
    return [c for c, pat in _PATTERNS.items() if pat.search(text or "")]


def text_evidence_posterior(p_image: np.ndarray, mentions: list[str], reliability: float = 0.8) -> np.ndarray:
    """Bayes update of the image posterior with the classes the text mentions.

    ``reliability`` = P(the text names the true class | it names a class).
    """
    if not mentions:
        return p_image
    k = len(CIFAR10_CLASSES)
    idx = [CIFAR10_CLASSES.index(m) for m in mentions]
    like = np.full(k, (1 - reliability) / (k - len(idx)))
    like[idx] = reliability / len(idx)
    post = p_image * like
    return post / post.sum()


@dataclass
class FusionReport:
    status: str                    # "confident" | "partial" | "conflict" | "defer"
    summary: str
    joint_confidence: float | None
    image: dict | None
    text: dict | None
    cross_modal: dict = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def fuse(img: SelfAssessment | None, txt: SelfAssessment | None, text: str = "",
         image_probs: np.ndarray | None = None, answer_threshold: float | None = None,
         reliability: float = 0.8) -> FusionReport:
    """Combine the two self-assessments into one report."""
    reasoning, cross = [], {}

    # ---- cross-modal evidence: the text may name what is in the picture
    if img is not None and image_probs is not None and text:
        mentions = mentioned_classes(text)
        cross["text_mentions"] = mentions
        if mentions and img.in_domain:
            post = text_evidence_posterior(image_probs, mentions, reliability)
            k_new = int(post.argmax())
            new_label, new_conf = CIFAR10_CLASSES[k_new], float(post[k_new])
            if img.label in mentions:
                reasoning.append(f"The text mentions '{img.label}', agreeing with the image "
                                 f"(confidence {img.confidence:.0%} → {new_conf:.0%}).")
            elif new_label != img.label:
                reasoning.append(f"The image alone leaned '{img.label}' ({img.confidence:.0%}), but the text "
                                 f"mentions {', '.join(mentions)}; combining both points to '{new_label}' ({new_conf:.0%}).")
                cross["revised_from"] = img.label
            else:
                cross["conflict"] = True
                reasoning.append(f"The text mentions {', '.join(mentions)}, but the image strongly says "
                                 f"'{img.label}'. The two inputs disagree.")
            img.extra["image_only"] = {"label": img.label, "confidence": img.confidence}
            img.label, img.confidence = new_label, new_conf
            if answer_threshold is not None:
                img.answer = bool(img.in_domain and new_conf >= answer_threshold)
            img.top = sorted(zip(CIFAR10_CLASSES, post.tolist()), key=lambda t: -t[1])[:3]

    parts = [a for a in (img, txt) if a is not None]
    answered = [a for a in parts if a.answer]
    for a in parts:
        if not a.answer:
            why = "; ".join(a.reasons) or "low confidence"
            reasoning.append(f"I won't commit to the {a.modality} prediction ('{a.label}'): {why}.")

    joint = float(np.prod([a.confidence for a in answered])) if answered else None
    desc = {"image": lambda a: f"a picture of a {a.label}", "text": lambda a: f"{a.label} sentiment"}

    if cross.get("conflict"):
        status = "conflict"
        summary = "The image and the text disagree about what is shown, so I'm flagging this for a human."
    elif not answered:
        status = "defer"
        summary = "I'm not confident enough in either input to give an answer."
    elif len(answered) < len(parts):
        status = "partial"
        a = answered[0]
        summary = f"I can only vouch for one part: {desc[a.modality](a)} ({a.confidence:.0%} sure)."
    else:
        status = "confident"
        if img is not None and txt is not None:
            summary = f"A {txt.label} post about a {img.label} ({joint:.0%} sure of both)."
        else:
            a = answered[0]
            summary = f"{desc[a.modality](a).capitalize()} ({a.confidence:.0%} sure)."

    return FusionReport(status, summary, joint, img.to_dict() if img else None,
                        txt.to_dict() if txt else None, cross, reasoning)


def legacy_fuse(image_data, text_data):
    """The original rule from the first version, kept for comparison in the evaluation.

    Picks whichever modality has the higher raw softmax confidence, defers if both
    are below 0.3 or within 0.05 of each other.
    """
    (il, ic), (tl, tc) = image_data, text_data
    if ic < 0.3 and tc < 0.3:
        return "Uncertain", max(ic, tc)
    if abs(ic - tc) < 0.05:
        return "Uncertain", max(ic, tc)
    return (il, ic) if ic > tc else (tl, tc)
