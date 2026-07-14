import pandas as pd

from Processing.analysis.action_briefs import build_action_briefs


def test_action_brief_answers_the_decision_questions():
    history = pd.DataFrame(
        {
            "task_id": ["T1", "T1", "T1"],
            "task_name": ["Civil handover"] * 3,
            "update_phase": ["Update_001", "Update_002", "Update_003"],
            "slip_days": [4, 7, 12],
            "is_critical": [True] * 3,
        }
    )
    forecasts = pd.DataFrame(
        {
            "task_id": ["T1"],
            "predicted_slip_days": [14],
            "forecast_confidence": [0.82],
            "low_confidence_flag": [False],
            "forecast_mae_days": [2.1],
            "confidence_reason": ["5 rolling tests; typical error 2.1 days"],
        }
    )

    result = build_action_briefs(history, forecasts)

    assert len(result) == 1
    brief = result.iloc[0]
    assert brief.priority == "P2"
    assert brief.owner_role == "Construction lead"
    assert "worsened by 5 days" in brief.what_changed
    assert "48-hour recovery review" in brief.recommended_action
    assert brief.human_gate == "Human approval required"
