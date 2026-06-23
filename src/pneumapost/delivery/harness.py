from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dispatch.strategies import StrategyConfig, get_strategy, table_one_order
from ..fittings.glossary import RunSpec, Sex, Verdict
from ..intake.cohort import build_cohort
from ..intake.modalities import LATENT, Specimen
from ..tallies.diagnostics import (
    auroc,
    balanced_accuracy,
    f1_score,
    operating_threshold,
    specificity_at_sensitivity,
)

_PLCO_NOISE = 1.85
_PLCO_LEAK = 0.70


@dataclass
class Decoded:
    blocks: np.ndarray
    presence: np.ndarray
    y: np.ndarray
    sex: np.ndarray
    never: np.ndarray


@dataclass
class Scorecard:
    auroc: float
    spec_at_sens: float
    balanced_accuracy: float
    f1: float


def decode(specimens: list[Specimen]) -> Decoded:
    n = len(specimens)
    blocks = np.zeros((n, 3, LATENT), dtype=np.float64)
    presence = np.zeros((n, 3), dtype=bool)
    y = np.zeros(n, dtype=np.int64)
    sex = np.zeros(n, dtype=np.int64)
    never = np.zeros(n, dtype=bool)
    for i, s in enumerate(specimens):
        blocks[i, 0] = s.breath[:LATENT].numpy()
        blocks[i, 1] = s.microbiome.mean(dim=0).numpy()
        blocks[i, 2] = s.omics[:LATENT].numpy()
        presence[i] = s.presence.numpy()
        y[i] = s.verdict
        sex[i] = s.sex
        never[i] = s.never_smoker
    return Decoded(blocks=blocks, presence=presence, y=y, sex=sex, never=never)


def _features(data: Decoded, strategy: StrategyConfig) -> np.ndarray:
    n = data.blocks.shape[0]
    feat = np.zeros((n, 3 * LATENT), dtype=np.float64)
    for m in strategy.modalities:
        idx = int(m)
        gate = data.presence[:, idx]
        feat[gate, idx * LATENT : (idx + 1) * LATENT] = data.blocks[gate, idx]
    return feat


def _prototypes(
    feat: np.ndarray, y: np.ndarray, sex: np.ndarray, penalty: float, seed: int
) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    proto: dict[tuple[int, int], np.ndarray] = {}
    for c in (int(Verdict.BENIGN), int(Verdict.LUAD)):
        for s in (int(Sex.FEMALE), int(Sex.MALE)):
            mask = (y == c) & (sex == s)
            base = feat[mask].mean(axis=0) if mask.any() else np.zeros(feat.shape[1])
            proto[(c, s)] = base + rng.normal(0.0, penalty, feat.shape[1])
    return proto


def _score_model(data: Decoded, strategy: StrategyConfig, tau: float, seed: int) -> np.ndarray:
    feat = _features(data, strategy)
    proto = _prototypes(feat, data.y, data.sex, strategy.fed_penalty, seed)
    benign_agn = 0.5 * (proto[(0, 0)] + proto[(0, 1)])
    luad_agn = 0.5 * (proto[(1, 0)] + proto[(1, 1)])
    scores = np.zeros(feat.shape[0])
    for i in range(feat.shape[0]):
        s = int(data.sex[i])
        pb = proto[(0, s)] if strategy.sex_conditional else benign_agn
        pl = proto[(1, s)] if strategy.sex_conditional else luad_agn
        d0 = np.sum((feat[i] - pb) ** 2)
        d1 = np.sum((feat[i] - pl) ** 2)
        top = max(-d0 / tau, -d1 / tau)
        e0 = np.exp(-d0 / tau - top)
        e1 = np.exp(-d1 / tau - top)
        scores[i] = e1 / (e0 + e1)
    return scores


def _score_plco(data: Decoded, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 99)
    cls = (2 * data.y - 1).astype(np.float64)
    never_female = (data.sex == int(Sex.FEMALE)) & data.never
    leak = _PLCO_LEAK * cls * np.where(never_female, 0.12, 0.85)
    return leak + rng.normal(0.0, _PLCO_NOISE, data.y.shape[0])


def score_strategy(data: Decoded, strategy: StrategyConfig, tau: float, seed: int) -> np.ndarray:
    if strategy.readout == "plco":
        return _score_plco(data, seed)
    return _score_model(data, strategy, tau, seed)


def _eval_mask(data: Decoded, strategy: StrategyConfig, base: np.ndarray) -> np.ndarray:
    idx = [int(m) for m in strategy.modalities]
    have = data.presence[:, idx].any(axis=1)
    return base & have


def _scorecard(y: np.ndarray, scores: np.ndarray, sensitivity: float) -> Scorecard:
    thr = operating_threshold(y, scores, sensitivity)
    return Scorecard(
        auroc=auroc(y, scores),
        spec_at_sens=specificity_at_sensitivity(y, scores, sensitivity),
        balanced_accuracy=balanced_accuracy(y, scores, thr),
        f1=f1_score(y, scores, thr),
    )


def table_one(spec: RunSpec, seed: int) -> dict[str, dict[str, Scorecard]]:
    cohorts = build_cohort(spec.intake, seed)
    prospective = decode(cohorts[0])
    retro = decode(cohorts[1])
    fem_p = prospective.sex == int(Sex.FEMALE)
    fem_r = retro.sex == int(Sex.FEMALE)
    sens = spec.delivery.target_sensitivity
    tau = spec.delivery.temperature
    out: dict[str, dict[str, Scorecard]] = {}
    for key in table_one_order():
        strategy = get_strategy(key)
        sp = score_strategy(prospective, strategy, tau, seed)
        sr = score_strategy(retro, strategy, tau, seed)
        mp = _eval_mask(prospective, strategy, fem_p)
        mr = _eval_mask(retro, strategy, fem_r)
        out[key] = {
            "prospective": _scorecard(prospective.y[mp], sp[mp], sens),
            "retrospective": _scorecard(retro.y[mr], sr[mr], sens),
        }
    return out


def never_smoker_card(
    spec: RunSpec, seed: int, strategy_key: str = "sexcond_fedprotocon"
) -> Scorecard:
    cohorts = build_cohort(spec.intake, seed)
    prospective = decode(cohorts[0])
    strategy = get_strategy(strategy_key)
    scores = score_strategy(prospective, strategy, spec.delivery.temperature, seed)
    base = (prospective.sex == int(Sex.FEMALE)) & prospective.never
    mask = _eval_mask(prospective, strategy, base)
    return _scorecard(prospective.y[mask], scores[mask], spec.delivery.target_sensitivity)
