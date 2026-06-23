from __future__ import annotations

import numpy as np
import torch

from ..fittings.glossary import IntakeSpec, Modality, Sex
from .modalities import LATENT, Specimen, empty_presence

FEMALE_AXIS = {Modality.BREATH: 0, Modality.MICROBIOME: 1, Modality.OMICS: 2}
SHARED_FEMALE_AXIS = 3
ESTROGEN_AXIS = 4
MALE_AXIS = {Modality.BREATH: 5, Modality.MICROBIOME: 6, Modality.OMICS: 7}
SEX_NUISANCE_AXIS = 8

_KAPPA = {Modality.BREATH: 0.90, Modality.MICROBIOME: 0.86, Modality.OMICS: 1.00}
_CLASS_GAP_FEMALE = 1.08
_CLASS_GAP_MALE = 0.80
_DECODE_NOISE = 1.60
_SHARED_FRACTION = 0.93
_NEVER_SMOKER_LIFT = 0.18
_ESTROGEN_GAP = 0.24
_SEX_NUISANCE = 0.70
_MICRO_TOKEN_NOISE = 0.30
_RETRO_OPTIMISM = 1.19


def _bernoulli(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(np.int64)


def _decoded(
    rng: np.random.Generator,
    modality: Modality,
    sex: np.ndarray,
    cls: np.ndarray,
    never: np.ndarray,
    driver_neg: np.ndarray,
    n: int,
    optimism: float,
) -> np.ndarray:
    v = rng.normal(0.0, _DECODE_NOISE, size=(n, LATENT)).astype(np.float32)
    kappa = _KAPPA[modality]
    female_mag = (
        optimism * kappa * (_CLASS_GAP_FEMALE + np.where(never == 1, _NEVER_SMOKER_LIFT, 0.0)) * cls
    )
    is_female = sex == int(Sex.FEMALE)
    v[:, FEMALE_AXIS[modality]] += np.where(is_female, (1.0 - _SHARED_FRACTION) * female_mag, 0.0)
    v[:, SHARED_FEMALE_AXIS] += np.where(is_female, _SHARED_FRACTION * female_mag, 0.0)
    v[:, MALE_AXIS[modality]] += np.where(sex == int(Sex.MALE), kappa * _CLASS_GAP_MALE * cls, 0.0)
    estro = is_female & (never == 1) & (driver_neg == 1)
    v[:, ESTROGEN_AXIS] += np.where(estro, kappa * _ESTROGEN_GAP * cls, 0.0)
    v[:, SEX_NUISANCE_AXIS] += _SEX_NUISANCE * (2 * sex - 1)
    return v


def _site_specimens(
    rng: np.random.Generator,
    spec: IntakeSpec,
    arm: int,
    site: int,
    n_female: int,
    n_male: int,
) -> list[Specimen]:
    n = n_female + n_male
    if n == 0:
        return []
    sex = np.concatenate([np.zeros(n_female, dtype=np.int64), np.ones(n_male, dtype=np.int64)])
    verdict = _bernoulli(rng, spec.luad_prevalence, n)
    never = _bernoulli(rng, spec.never_smoker_share, n)
    driver_neg = _bernoulli(rng, spec.signal.driver_negative_share, n)
    cls = (2 * verdict - 1).astype(np.float32)
    optimism = _RETRO_OPTIMISM if arm == 1 else 1.0

    decoded = {
        m: _decoded(rng, m, sex, cls, never, driver_neg, n, optimism)
        for m in (Modality.BREATH, Modality.MICROBIOME, Modality.OMICS)
    }

    breath = rng.normal(0.0, 1.0, size=(n, spec.breath_len)).astype(np.float32)
    breath[:, :LATENT] = decoded[Modality.BREATH]
    omics = rng.normal(0.0, 1.0, size=(n, spec.omics_dim)).astype(np.float32)
    omics[:, :LATENT] = decoded[Modality.OMICS]
    micro = (
        decoded[Modality.MICROBIOME][:, None, :]
        + rng.normal(0.0, _MICRO_TOKEN_NOISE, size=(n, spec.taxa_set_size, LATENT))
    ).astype(np.float32)

    keep = np.stack(
        [
            rng.random(n) >= spec.missing_breath,
            rng.random(n) >= spec.missing_microbiome,
            rng.random(n) >= spec.missing_omics,
        ],
        axis=1,
    )
    keep[~keep.any(axis=1), 0] = True

    out: list[Specimen] = []
    for i in range(n):
        presence = empty_presence()
        presence[Modality.BREATH] = bool(keep[i, 0])
        presence[Modality.MICROBIOME] = bool(keep[i, 1])
        presence[Modality.OMICS] = bool(keep[i, 2])
        out.append(
            Specimen(
                breath=torch.from_numpy(breath[i]),
                microbiome=torch.from_numpy(micro[i]),
                omics=torch.from_numpy(omics[i]),
                presence=presence,
                verdict=int(verdict[i]),
                sex=int(sex[i]),
                never_smoker=bool(never[i]),
                driver_negative=bool(driver_neg[i]),
                arm=arm,
                site=site,
            )
        )
    return out


def build_cohort(spec: IntakeSpec, seed: int) -> dict[int, list[Specimen]]:
    cohorts: dict[int, list[Specimen]] = {}
    arms = [
        (spec.prospective_female, spec.prospective_male),
        (spec.retro_female, spec.retro_male),
    ]
    for arm, (fem, male) in enumerate(arms):
        collected: list[Specimen] = []
        for site in range(spec.n_sites):
            rng = np.random.default_rng((seed + 1) * 1009 + arm * 131 + site)
            collected.extend(
                _site_specimens(
                    rng,
                    spec,
                    arm=arm,
                    site=site,
                    n_female=fem[site] if site < len(fem) else 0,
                    n_male=male[site] if site < len(male) else 0,
                )
            )
        cohorts[arm] = collected
    return cohorts


def by_site(specimens: list[Specimen], n_sites: int) -> list[list[Specimen]]:
    buckets: list[list[Specimen]] = [[] for _ in range(n_sites)]
    for s in specimens:
        buckets[s.site].append(s)
    return buckets
