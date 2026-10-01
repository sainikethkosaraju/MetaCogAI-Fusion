"""End-to-end pipeline: experts -> metacognition -> fusion."""
from __future__ import annotations

from functools import cached_property

import numpy as np
from PIL import Image

from .experts import CIFAR10_CLASSES, SENTIMENTS, ImageExpert, TextExpert
from .fusion import FusionReport, fuse
from .metacognition import Metacognition, softmax


class MetaCogAI:
    """Load once, then call :meth:`analyze` with an image, a text, or both."""

    def __init__(self, mc_samples: int = 20):
        self.meta = Metacognition.load()
        self.mc_samples = mc_samples

    @cached_property
    def image_expert(self) -> ImageExpert:
        return ImageExpert()

    @cached_property
    def text_expert(self) -> TextExpert:
        return TextExpert()

    def analyze(self, image: Image.Image | None = None, text: str | None = None) -> FusionReport:
        if image is None and not (text and text.strip()):
            raise ValueError("Provide an image, a text, or both.")
        img_a = txt_a = probs = None
        if image is not None:
            feats, logits = self.image_expert.from_pil(image)
            img_a = self.meta.assess_image(logits, feats, CIFAR10_CLASSES)
            probs = softmax(logits, self.meta.cfg["image"]["temperature"])
        if text and text.strip():
            logits = self.text_expert.logits([text])[0].numpy()
            mc = self.text_expert.mc_dropout_probs(text, self.mc_samples).numpy() if self.mc_samples else None
            txt_a = self.meta.assess_text(logits, SENTIMENTS, mc)
        return fuse(img_a, txt_a, text or "", probs,
                    answer_threshold=self.meta.cfg["image"]["answer_threshold"],
                    reliability=self.meta.cfg["fusion"]["mention_reliability"])
