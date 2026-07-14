"""Convert schedule evidence into reviewable, owned decision prompts."""

from __future__ import annotations

import re

import pandas as pd


def _phase_number(value: object) -> int:
    match = re.search(r"(\d+)(?!.*\d)", str(value))
    return int(match.group(1)) if match else 0


def _owner_for(task_name: str) -> str:
    name = task_name.lower()
    if any(word in name for word in ("design", "approval", "review")):
        return "Design authority"
    if any(word in name for word in ("test", "commission", "assurance")):
        return "Test manager"
    if any(word in name for word in ("supplier", "procure", "package")):
        return "Commercial lead"
    if any(word in name for word in ("access", "civil", "construct")):
        return "Construction lead"
    return "Planning lead"


def build_action_briefs(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Build a ranked queue answering what changed, why it matters and what to do.

    Recommendations are deliberately framed as decision prompts. They preserve a
    human approval gate and never imply that a model can diagnose an operational
    cause that is absent from the supplied evidence.
    """

    if history is None or history.empty or forecasts is None or forecasts.empty:
        return pd.DataFrame()
    required = {"task_id", "update_phase", "slip_days"}
    if not required.issubset(history.columns):
        return pd.DataFrame()

    work = history.copy()
    work["phase_order"] = work.update_phase.map(_phase_number)
    work["slip_days"] = pd.to_numeric(work.slip_days, errors="coerce").fillna(0)
    work = work.sort_values(["task_id", "phase_order"])
    forecasts_by_task = forecasts.set_index("task_id")
    briefs = []

    for task_id, task in work.groupby("task_id"):
        if task_id not in forecasts_by_task.index:
            continue
        task = task.tail(3)
        latest = task.iloc[-1]
        previous = task.iloc[-2] if len(task) > 1 else latest
        forecast = forecasts_by_task.loc[task_id]
        if isinstance(forecast, pd.DataFrame):
            forecast = forecast.iloc[0]

        task_name = str(latest.get("task_name", f"Task {task_id}"))
        current_slip = float(latest.slip_days)
        movement = current_slip - float(previous.slip_days)
        predicted = float(forecast.get("predicted_slip_days", current_slip))
        is_critical = bool(latest.get("is_critical", False))
        low_confidence = bool(forecast.get("low_confidence_flag", True))
        owner = _owner_for(task_name)

        if movement >= 2:
            changed = f"Slippage worsened by {movement:.0f} days in the latest update to {current_slip:.0f} days."
        elif movement <= -2:
            changed = f"The latest update recovered {abs(movement):.0f} days, leaving {current_slip:.0f} days of slippage."
        else:
            changed = f"Slippage held at {current_slip:.0f} days in the latest update."

        if low_confidence:
            action = "Validate the finish-date evidence and dependency logic before approving a recovery commitment."
            next_evidence = "A dated owner update, confirmed predecessor logic and one further reporting observation."
            question = "Is the evidence strong enough to commit to a recovery date?"
        elif is_critical and predicted > 7:
            action = "Run a 48-hour recovery review, name the blocked dependency and bring one costed recovery option for approval."
            next_evidence = "Named blocker, accountable owner, decision date and quantified recovery mechanism."
            question = "Which owned intervention will protect the critical path?"
        elif predicted > 14:
            action = "Choose between an owned recovery action and a controlled reforecast at the next control gate."
            next_evidence = "Recovery logic with dates, resource commitment and affected successor activities."
            question = "Do we recover this date or formally reforecast it?"
        else:
            action = "Keep under exception review and confirm that the next update does not introduce new movement."
            next_evidence = "Next accepted schedule update and confirmation of the current constraint."
            question = "Can this remain on watch without director intervention?"

        score = (
            min(max(predicted, 0), 60) * 0.8
            + max(movement, 0) * 1.5
            + (18 if is_critical else 0)
            + (8 if low_confidence else 0)
        )
        # Keep P1 deliberately scarce so the queue remains usable in a control meeting.
        priority = "P1" if score >= 55 else "P2" if score >= 25 else "P3"
        why = (
            f"The selected model expects {predicted:.0f} days of slippage. "
            f"Its rolling-test MAE is {float(forecast.get('forecast_mae_days', 0)):.1f} days"
            f"{' and the activity is marked critical' if is_critical else ''}."
        )
        briefs.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "priority": priority,
                "priority_score": round(score, 1),
                "decision_question": f"{task_name}: {question[0].lower()}{question[1:]}",
                "what_changed": changed,
                "forecast_if_no_action": f"Next reported position: about {predicted:.0f} days late.",
                "why_now": why,
                "recommended_action": action,
                "owner_role": owner,
                "decision_by": "Before the next control gate" if priority != "P3" else "At the next exception review",
                "evidence": f"{previous.update_phase} to {latest.update_phase}; {forecast.get('confidence_reason', 'validation unavailable')}",
                "confidence": round(float(forecast.get("forecast_confidence", 0)), 3),
                "next_evidence": next_evidence,
                "human_gate": "Human approval required",
            }
        )

    result = pd.DataFrame(briefs).sort_values(
        ["priority_score", "confidence"], ascending=[False, False]
    )
    return result.head(top_n).reset_index(drop=True) if top_n else result.reset_index(drop=True)
