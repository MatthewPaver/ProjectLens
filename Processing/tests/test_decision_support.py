import pytest

from Processing.analysis.decision_support import (
    Intervention,
    RiskDriver,
    compare_interventions,
    rank_root_causes,
    simulate_completion_risk,
)


DRIVERS = [
    RiskDriver("handover", "Civil handover", "Delivery", 0.82, 5, 12, 22, 1.0),
    RiskDriver("approval", "Design approvals", "Governance", 0.66, 2, 8, 16, 0.85),
    RiskDriver("testing", "Testing capacity", "Resource", 0.58, 3, 7, 14, 0.9),
]

INTERVENTIONS = [
    Intervention(
        "resequence",
        "Resequence commissioning",
        "Decouple testing readiness from final civil handover.",
        "Planning lead",
        2,
        "Low",
        {"handover": 0.45, "testing": 0.2},
    ),
    Intervention(
        "night-shift",
        "Add a night shift",
        "Increase specialist testing capacity for four weeks.",
        "Delivery director",
        4,
        "High",
        {"testing": 0.65},
    ),
]


def test_simulation_is_repeatable_and_ordered():
    first = simulate_completion_risk(DRIVERS, contingency_days=18, seed=42)
    second = simulate_completion_risk(DRIVERS, contingency_days=18, seed=42)

    assert first == second
    assert first["p10_days"] <= first["p50_days"] <= first["p80_days"] <= first["p90_days"]
    assert 0 <= first["on_time_probability"] <= 100


def test_intervention_improves_the_same_baseline_draws():
    comparison = compare_interventions(
        DRIVERS,
        INTERVENTIONS,
        contingency_days=18,
        iterations=4_000,
        seed=7,
    )

    for scenario in comparison["scenarios"]:
        assert scenario["p80_recovery_days"] >= 0
        assert scenario["confidence_uplift"] >= 0
        assert scenario["result"]["p80_days"] <= comparison["baseline"]["p80_days"]


def test_root_causes_are_ranked_by_expected_contribution():
    ranked = rank_root_causes(DRIVERS)
    assert ranked[0]["id"] == "handover"
    assert ranked[0]["expected_days"] >= ranked[-1]["expected_days"]


def test_invalid_risk_inputs_fail_loudly():
    invalid = [RiskDriver("bad", "Bad input", "Data", 1.2, 1, 2, 3)]
    with pytest.raises(ValueError, match="probability"):
        simulate_completion_risk(invalid)

