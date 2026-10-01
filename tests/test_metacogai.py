import numpy as np
import pytest

from metacogai.experts import CIFAR10_CLASSES, SENTIMENTS
from metacogai.fusion import fuse, legacy_fuse, mentioned_classes, text_evidence_posterior
from metacogai.metacognition import (Metacognition, SelfAssessment, ece, energy, entropy,
                                     fit_temperature, selective_threshold, softmax)


# ------------------------------------------------------------- pure maths
def test_softmax_and_entropy():
    p = softmax(np.array([[0.0, 0.0], [10.0, -10.0]]))
    assert np.allclose(p.sum(1), 1)
    assert entropy(p)[0] == pytest.approx(1.0) and entropy(p)[1] < 0.01


def test_energy_is_logsumexp():
    z = np.array([[1.0, 2.0, 3.0]])
    assert energy(z)[0] == pytest.approx(np.log(np.exp(z).sum()))


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 3)) * 1.5
    y = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
    T = fit_temperature(true_logits * 3.0, y)          # model 3x over-confident
    assert 2.4 < T < 3.6


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1, 20000)
    correct = (rng.random(20000) < conf).astype(float)
    assert ece(conf, correct) < 0.02


def test_selective_threshold_meets_target():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0, 1, 5000)
    correct = (rng.random(5000) < conf).astype(float)
    thr = selective_threshold(conf, correct, 0.9)
    assert correct[conf >= thr].mean() >= 0.9
    assert 0 < (conf >= thr).mean() < 1


# ------------------------------------------------------------- fusion
def test_class_mentions():
    assert mentioned_classes("My puppy chased a boat!") == ["dog", "ship"]
    assert mentioned_classes("Cars and trucks") == ["automobile", "truck"]
    assert mentioned_classes("scattered thoughts") == []        # 'cat' inside a word is not a mention


def test_text_evidence_can_flip_an_ambiguous_image():
    p = np.full(10, 0.02)
    p[CIFAR10_CLASSES.index("cat")], p[CIFAR10_CLASSES.index("dog")] = 0.45, 0.39
    post = text_evidence_posterior(p / p.sum(), ["dog"], reliability=0.8)
    assert CIFAR10_CLASSES[post.argmax()] == "dog"


def _assess(mod, label, conf, answer=True, in_domain=True):
    return SelfAssessment(mod, label, conf, conf, 0.1, in_domain, answer)


def test_fusion_statuses():
    assert fuse(_assess("image", "cat", .97), _assess("text", "positive", .99)).status == "confident"
    assert fuse(_assess("image", "cat", .5, answer=False), _assess("text", "positive", .99)).status == "partial"
    assert fuse(_assess("image", "cat", .5, False), _assess("text", "positive", .55, False)).status == "defer"


def test_fusion_reports_conflict():
    p = np.full(10, 0.001)
    p[CIFAR10_CLASSES.index("ship")] = 1.0
    p /= p.sum()
    r = fuse(_assess("image", "ship", float(p.max())), _assess("text", "positive", .99),
             text="my lovely cat", image_probs=p)
    assert r.status == "conflict"


def test_legacy_rule_discards_a_modality():
    assert legacy_fuse(("cat", 0.9), ("positive", 0.6)) == ("cat", 0.9)
    assert legacy_fuse(("cat", 0.2), ("positive", 0.1))[0] == "Uncertain"


# ------------------------------------------------------------- metacognition config
def test_assessments_with_shipped_calibration():
    meta = Metacognition.load()
    z = np.zeros(10)
    z[3] = 12.0                                         # very confident "cat"
    a = meta.assess_image(z, None, CIFAR10_CLASSES)
    assert a.label == "cat" and a.confidence > 0.9
    weak = meta.assess_image(np.zeros(10), None, CIFAR10_CLASSES)   # flat logits
    assert not weak.answer
    t = meta.assess_text(np.array([-3.0, 3.0]), SENTIMENTS)
    assert t.label == "positive"
    flip = meta.assess_text(np.array([-3.0, 3.0]), SENTIMENTS, mc_probs=np.array([[0.9, 0.1]] * 10))
    assert not flip.answer                              # unstable under dropout -> abstain


# ------------------------------------------------------------- end to end (slow: loads both models)
@pytest.mark.slow
def test_end_to_end_examples():
    from pathlib import Path

    from PIL import Image

    from metacogai.system import MetaCogAI
    ex = Path(__file__).resolve().parent.parent / "static/examples"
    sys = MetaCogAI(mc_samples=5)
    r = sys.analyze(Image.open(ex / "cifar_cat.png"), "Took this photo of my kitten, she is wonderful!")
    assert r.image["label"] == "cat" and r.text["label"] == "positive" and r.status == "confident"
    r = sys.analyze(Image.open(ex / "ood_noise.png"))
    assert not r.image["in_domain"] and r.status == "defer"


@pytest.mark.slow
def test_api_roundtrip():
    from pathlib import Path

    from fastapi.testclient import TestClient

    from app import app
    ex = Path(__file__).resolve().parent.parent / "static/examples"
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    with open(ex / "cifar_ambiguous.png", "rb") as f:
        r = client.post("/api/analyze", files={"image": ("x.png", f, "image/png")},
                        data={"text": "Took this photo of my kitten on the porch, she is the best!"})
    assert r.status_code == 200
    body = r.json()
    assert body["image"]["label"] == "cat" and body["cross_modal"].get("revised_from") == "dog"
    assert client.post("/api/analyze", data={"text": ""}).status_code == 400
    bad = client.post("/api/analyze", files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
