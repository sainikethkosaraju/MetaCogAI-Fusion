"""The two perception models ("experts").

* Image expert: ResNet-18 trained from scratch on CIFAR-10 (32x32, 10 classes).
* Text expert: DistilBERT fine-tuned on IMDb movie reviews (positive/negative).

Each expert only returns **logits**. Turning logits into a decision -- and
deciding whether that decision can be trusted -- is the job of
``metacognition.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
SENTIMENTS = ("negative", "positive")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageExpert:
    """CIFAR-10 classifier. Arbitrary photos are centre-cropped to a square and
    resized to 32x32 with the exact normalisation used during training --
    getting this wrong (as an earlier version did) drops accuracy from 84% to 58%."""

    labels = CIFAR10_CLASSES

    def __init__(self, weights: Path = WEIGHTS / "image_resnet18_cifar10.pth"):
        self.device = _device()
        self.model = models.resnet18(num_classes=10)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    @staticmethod
    def preprocess(img: Image.Image) -> torch.Tensor:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = ImageOps.fit(img, (32, 32), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - 0.5) / 0.5

    @torch.no_grad()
    def features_and_logits(self, batch: torch.Tensor):
        """``batch``: [B, 3, 32, 32] normalised to [-1, 1].

        Returns (list of 4 globally pooled feature maps, one per ResNet stage --
        [B, 64], [B, 128], [B, 256], [B, 512] -- and logits [B, 10]).
        """
        m = self.model
        feats, logits = [[], [], [], []], []
        for i in range(0, len(batch), 256):
            h = m.maxpool(m.relu(m.bn1(m.conv1(batch[i:i + 256].to(self.device)))))
            for j, stage in enumerate((m.layer1, m.layer2, m.layer3, m.layer4)):
                h = stage(h)
                feats[j].append(h.mean((2, 3)).cpu())
            logits.append(m.fc(feats[3][-1].to(self.device)).cpu())
        return [torch.cat(f) for f in feats], torch.cat(logits)

    def logits(self, batch: torch.Tensor) -> torch.Tensor:
        return self.features_and_logits(batch)[1]

    def from_pil(self, img: Image.Image):
        f, z = self.features_and_logits(self.preprocess(img).unsqueeze(0))
        return [x[0].numpy() for x in f], z[0].numpy()


class TextExpert:
    """IMDb sentiment classifier (DistilBERT)."""

    labels = SENTIMENTS

    def __init__(self, path: Path = WEIGHTS / "text_model", max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        self.model.to(self.device).eval()
        self.max_length = max_length

    def _encode(self, texts):
        return self.tokenizer(list(texts), truncation=True, padding=True,
                              max_length=self.max_length, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def logits(self, texts, batch_size: int = 16) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self.model(**self._encode(texts[i:i + batch_size])).logits.cpu())
        return torch.cat(out)

    @torch.no_grad()
    def mc_dropout_probs(self, text: str, samples: int = 20) -> torch.Tensor:
        """[samples, 2] class probabilities with dropout left on (Gal & Ghahramani, 2016)."""
        enc = self._encode([text])
        self.model.train()
        try:
            return torch.stack([self.model(**enc).logits[0].softmax(-1).cpu() for _ in range(samples)])
        finally:
            self.model.eval()
