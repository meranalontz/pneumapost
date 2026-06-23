from __future__ import annotations

import numpy as np
import torch

from pneumapost.apparatus.network import build_network
from pneumapost.delivery.calibrate import calibrate_threshold
from pneumapost.delivery.inference import decide, infer, sex_matched_probability
from pneumapost.fittings.glossary import RunSpec
from pneumapost.intake.cohort import build_cohort
from pneumapost.intake.modalities import collate


def test_probabilities_are_bounded(tiny_spec: RunSpec) -> None:
    torch.manual_seed(0)
    net = build_network(tiny_spec.apparatus, tiny_spec.intake.omics_dim)
    batch = collate(build_cohort(tiny_spec.intake, 0)[0][:20])
    p = infer(net, batch, tiny_spec.apparatus.tau)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def test_decision_threshold_is_monotone() -> None:
    p = torch.tensor([0.1, 0.4, 0.6, 0.9])
    high = decide(p, 0.8)
    low = decide(p, 0.3)
    assert int(low.sum()) >= int(high.sum())


def test_calibrated_threshold_reaches_target_sensitivity() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    prob = np.clip(rng.normal(0.5 + 0.25 * y, 0.1), 0, 1)
    thr = calibrate_threshold(y, prob, 0.95)
    achieved = float((prob[y == 1] >= thr).mean())
    assert achieved >= 0.95


def test_sex_matched_probability_uses_correct_anchor() -> None:
    dim = 8
    bank = torch.zeros(2, 2, dim)
    bank[1, 0, 0] = 1.0
    z = torch.zeros(1, dim)
    z[0, 0] = 1.0
    p_female = sex_matched_probability(z, bank, torch.tensor([0]), tau=0.2)
    p_male = sex_matched_probability(z, bank, torch.tensor([1]), tau=0.2)
    assert float(p_female) > float(p_male)
