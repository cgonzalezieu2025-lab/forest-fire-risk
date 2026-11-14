import numpy as np
import torch

from forest_fire_risk import cnn


def test_prep_and_predict_shapes():
    cnn.DEVICE = "cpu"
    X = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    aug = cnn.prep(X, train=True)
    assert aug.shape == (4, 3, 64, 64) and torch.isfinite(aug).all()
    p = cnn.predict(cnn.make_model(pretrained=False), X)
    assert p.shape == (4,) and np.all((p >= 0) & (p <= 1))


def test_freeze_keeps_early_blocks_fixed():
    trainable = lambda k: sum(w.requires_grad for w in cnn.make_model(pretrained=False, freeze=k).parameters())
    assert trainable(0) > trainable(1) > trainable(3) >= 2          # the new last layer always trains
