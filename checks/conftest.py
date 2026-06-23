from __future__ import annotations

import dataclasses

import pytest

from pneumapost.fittings.glossary import IntakeSpec, RunSpec


@pytest.fixture(scope="session")
def main_spec() -> RunSpec:
    return RunSpec()


@pytest.fixture(scope="session")
def tiny_spec() -> RunSpec:
    intake = dataclasses.replace(
        IntakeSpec(),
        prospective_female=(40, 36, 30, 24),
        prospective_male=(8, 6, 6, 4),
        retro_female=(30, 26, 22, 18),
        retro_male=(6, 4, 4, 4),
    )
    return dataclasses.replace(RunSpec(), intake=intake)
