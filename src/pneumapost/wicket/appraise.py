from __future__ import annotations

from ..delivery.harness import decode, never_smoker_card, score_strategy, table_one
from ..dispatch.strategies import get_strategy, table_one_order
from ..fittings.clerking import get_logger
from ..fittings.glossary import Sex
from ..fittings.hocon_io import load_circular
from ..intake.cohort import build_cohort
from ..tallies.inference_stats import bootstrap_ci, delong_test

_log = get_logger("wicket.appraise")


def appraise(config: str, *override: str, arm: str = "prospective") -> None:
    spec = load_circular(config, override)
    table = table_one(spec, spec.seed)
    _log.info("%-26s %7s %7s %7s %7s", "method", "AUROC", "Spec95", "BalAcc", "F1")
    for key in table_one_order():
        card = table[key][arm]
        _log.info(
            "%-26s %7.3f %7.3f %7.3f %7.3f",
            get_strategy(key).label,
            card.auroc,
            card.spec_at_sens,
            card.balanced_accuracy,
            card.f1,
        )
    ns = never_smoker_card(spec, spec.seed)
    _log.info("never-smoker female AUROC %.3f", ns.auroc)

    cohorts = build_cohort(spec.intake, spec.seed)
    data = decode(cohorts[0])
    fem = data.sex == int(Sex.FEMALE)
    proposed = score_strategy(
        data, get_strategy("sexcond_fedprotocon"), spec.delivery.temperature, spec.seed
    )
    rival = score_strategy(data, get_strategy("dflmp"), spec.delivery.temperature, spec.seed)
    z, p = delong_test(data.y[fem], proposed[fem], rival[fem])
    lo, hi = bootstrap_ci(data.y[fem], proposed[fem], spec.delivery.bootstrap, 0.05, spec.seed)
    _log.info("DeLong proposed vs DFL-MP: z=%.3f p=%.4f", z, p)
    _log.info("proposed AUROC 95%% CI [%.3f, %.3f]", lo, hi)
