"""Transparent scenario modelling for ProjectLens decision support.

The module intentionally keeps the model small and inspectable. It is designed to
compare interventions using common random numbers, not to claim certainty about a
project completion date. Production use requires organisation-specific calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from statistics import mean
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RiskDriver:
    id: str
    name: str
    category: str
    probability: float
    minimum_days: float
    most_likely_days: float
    maximum_days: float
    criticality: float = 1.0
    evidence: str = ""

    def validate(self) -> None:
        if not 0 <= self.probability <= 1:
            raise ValueError(f"{self.id}: probability must be between 0 and 1")
        if not 0 <= self.criticality <= 1:
            raise ValueError(f"{self.id}: criticality must be between 0 and 1")
        if not self.minimum_days <= self.most_likely_days <= self.maximum_days:
            raise ValueError(f"{self.id}: triangular estimates are out of order")


@dataclass(frozen=True)
class Intervention:
    id: str
    name: str
    description: str
    owner_role: str
    effort: int
    cost_band: str
    reductions: Mapping[str, float]

    def validate(self) -> None:
        if not 1 <= self.effort <= 5:
            raise ValueError(f"{self.id}: effort must be between 1 and 5")
        for driver_id, reduction in self.reductions.items():
            if not 0 <= reduction <= 1:
                raise ValueError(
                    f"{self.id}: reduction for {driver_id} must be between 0 and 1"
                )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def simulate_completion_risk(
    drivers: Iterable[RiskDriver],
    *,
    reductions: Mapping[str, float] | None = None,
    committed_delay_days: float = 0,
    contingency_days: float = 0,
    iterations: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Simulate residual completion delay from independent risk drivers.

    ``reductions`` represents the proportion of each driver's sampled impact that
    an intervention is expected to remove. Using a shared seed makes scenario
    differences attributable to the intervention instead of simulation noise.
    """

    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if contingency_days < 0:
        raise ValueError("contingency_days cannot be negative")

    driver_list = list(drivers)
    for driver in driver_list:
        driver.validate()

    reductions = reductions or {}
    rng = random.Random(seed)
    outcomes: list[float] = []

    for _ in range(iterations):
        delay = committed_delay_days
        for driver in driver_list:
            if rng.random() > driver.probability:
                continue
            sampled_impact = rng.triangular(
                driver.minimum_days,
                driver.maximum_days,
                driver.most_likely_days,
            )
            reduction = reductions.get(driver.id, 0)
            delay += sampled_impact * driver.criticality * (1 - reduction)
        outcomes.append(max(0, delay))

    on_time = sum(value <= contingency_days for value in outcomes) / iterations
    return {
        "p10_days": round(_percentile(outcomes, 0.10), 1),
        "p50_days": round(_percentile(outcomes, 0.50), 1),
        "p80_days": round(_percentile(outcomes, 0.80), 1),
        "p90_days": round(_percentile(outcomes, 0.90), 1),
        "mean_days": round(mean(outcomes), 1),
        "on_time_probability": round(on_time * 100, 1),
        "iterations": iterations,
    }


def rank_root_causes(drivers: Iterable[RiskDriver]) -> list[dict[str, object]]:
    """Rank drivers by transparent expected critical-path contribution."""

    ranked = []
    for driver in drivers:
        driver.validate()
        expected_days = driver.probability * driver.most_likely_days * driver.criticality
        ranked.append({**asdict(driver), "expected_days": round(expected_days, 1)})
    return sorted(ranked, key=lambda item: item["expected_days"], reverse=True)


def compare_interventions(
    drivers: Iterable[RiskDriver],
    interventions: Iterable[Intervention],
    *,
    committed_delay_days: float = 0,
    contingency_days: float = 0,
    iterations: int = 10_000,
    seed: int = 2026,
) -> dict[str, object]:
    """Compare intervention outcomes against a common baseline."""

    driver_list = list(drivers)
    baseline = simulate_completion_risk(
        driver_list,
        committed_delay_days=committed_delay_days,
        contingency_days=contingency_days,
        iterations=iterations,
        seed=seed,
    )

    scenarios = []
    for intervention in interventions:
        intervention.validate()
        result = simulate_completion_risk(
            driver_list,
            reductions=intervention.reductions,
            committed_delay_days=committed_delay_days,
            contingency_days=contingency_days,
            iterations=iterations,
            seed=seed,
        )
        p80_recovery = round(float(baseline["p80_days"]) - float(result["p80_days"]), 1)
        confidence_uplift = round(
            float(result["on_time_probability"])
            - float(baseline["on_time_probability"]),
            1,
        )
        scenarios.append(
            {
                **asdict(intervention),
                "result": result,
                "p80_recovery_days": p80_recovery,
                "confidence_uplift": confidence_uplift,
                "value_score": round((p80_recovery + confidence_uplift / 4) / intervention.effort, 1),
            }
        )

    scenarios.sort(key=lambda item: item["value_score"], reverse=True)
    return {"baseline": baseline, "scenarios": scenarios}

