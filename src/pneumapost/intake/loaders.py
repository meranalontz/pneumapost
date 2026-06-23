from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from ..fittings.glossary import IntakeSpec
from .cohort import build_cohort, by_site
from .modalities import Batch, Specimen, collate


class SpecimenSet(Dataset[Specimen]):
    def __init__(self, specimens: list[Specimen]) -> None:
        self._items = specimens

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Specimen:
        return self._items[index]


def make_loader(
    specimens: list[Specimen],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[Specimen]:
    from ..fittings.seed import spawn_generator

    return DataLoader(
        SpecimenSet(specimens),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate,
        generator=spawn_generator(seed) if shuffle else None,
    )


def site_loaders(
    spec: IntakeSpec,
    batch_size: int,
    seed: int,
) -> tuple[list[DataLoader[Specimen]], list[Specimen], list[Specimen]]:
    cohorts = build_cohort(spec, seed)
    prospective = cohorts[0]
    retro = cohorts[1]
    buckets = by_site(prospective, spec.n_sites)
    loaders = [
        make_loader(b, batch_size, shuffle=True, seed=seed + k) for k, b in enumerate(buckets)
    ]
    return loaders, prospective, retro


def full_batch(specimens: list[Specimen]) -> Batch:
    return collate(specimens)
