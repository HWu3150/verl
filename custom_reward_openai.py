"""Custom reward function that calls an OpenAI-style chat API.

Place this file anywhere readable by the trainer, then point
`reward.custom_reward_function.path` to its absolute path. Example:

    python -m verl.trainer.ppo_trainer \
      reward.custom_reward_function.path=/home/you/verl/custom_reward_openai.py \
      reward.custom_reward_function.reward_kwargs.api_key=$OPENAI_API_KEY

Environment variables:
  - OPENAI_API_KEY: used if api_key arg is not provided.
  - OPENAI_BASE_URL: optional; overrides default API base.

Returned value must be float or a dict containing "score"; verl's
`NaiveRewardManager` will place it in rm_scores for GRPO/PPO.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from openai import OpenAI

PROMPT_TEMPLATE = """You are an expert hindsight judge for action-observation trajectories.

You will be given:
(2) The FULL trajectory: initial observation, task, steps 1..T, and outcome.
(3) The student rater's output for step t: exactly two lines:
    Reason: ...
    Label: negative|neutral|positive

**YOUR JOB**
Using the FULL trajectory as hindsight evidence, evaluate whether the student's LABEL for step t is correct under the label meanings below.
Then assign a scalar reward for training the student.

**LABEL MEANINGS (hindsight)**
- positive: step t clearly contributes to eventual task completion OR gathers information/resources that are later used to make progress.
- neutral: step t is neither clearly helpful nor harmful in hindsight; it does not materially change the chance of success.
- negative: step t is unproductive/harmful in hindsight: wasted steps, regressions, contradictions, loops, or actions that reduce success probability.

**JUDGING RULES (important)**
- First decide the hindsight gold_label for step t using the FULL trajectory (you may use events after step t).
- Reward is label-dominant: score LABEL correctness first; REASON only gives a small adjustment when the label is not clearly wrong.
- REASON is judged only by evidence alignment: it must match facts in the FULL trajectory and support the label.
  - If the reason hallucinates/contradicts the trajectory, apply a negative adjustment.
  - If the reason is generic but not wrong, give zero adjustment.
- Do NOT reward verbosity or style.
- If the student's output format/label is invalid (not exactly two lines or label not in {{negative, neutral, positive}}), set reward = -1.0 and stop.
- If ambiguous, prefer gold_label = neutral and keep reward near 0.

**REWARD SCHEME (two-stage: label first, reason as small adjustment)**

Step A) Decide the hindsight gold_label for step t, using the FULL trajectory.

Step B) Compute base reward from student's label vs gold_label:
- exact match: base = +0.8
- adjacent mismatch (positive<->neutral or neutral<->negative): base = +0.2
- opposite mismatch (positive<->negative): base = -0.8
- invalid format or invalid label: reward = -1.0 and stop.

Step C) Reason adjustment (ONLY if base >= +0.2):
Evaluate whether the student's Reason is helpful with the task progress and actually supports the given label.
Set reason_adj in {{+0.2, 0.0, -0.2}}:
- +0.2: correctly identifies helpful/harmful aspects of the trajectory, showing good understanding.
-  0.0: generic/weak but not wrong
- -0.2: contradicts the trajectory, hallucinates, or completely misses the point

Final reward = clip(base + reason_adj, -1.0, +1.0)

IMPORTANT:
- Do not reward verbosity or style.
- Reason adjustment must be small; label correctness dominates.

Also output:
- label_match: one of {{"exact","adjacent","opposite","invalid"}}
- reason_adj: -0.2|0|+0.2
- confidence: 0.0..1.0 for your gold_label certainty

FULL trajectory (steps 1..T):
{full_traj}

We are judging step t={t}.
Student output (two lines):
{student_two_lines}

OUTPUT FORMAT (valid JSON, no extra text)
{{
  "gold_label": "negative|neutral|positive",
  "reward": <float>,
  "confidence": <float>,
  "brief_feedback": "<1~2 sentence>"
}}
"""


def _build_prompt(solution_str: str, reference_answer: str, extra_info: Dict[str, Any] | None) -> str:
    # Deprecated signature; kept for backward compatibility
    raise NotImplementedError


def compute_score(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: str | None = None,  # unused; kept for API compatibility
    extra_info: Dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float | None = 30.0,
    base_url: str | None = None,
) -> float | Dict[str, Any]:
    """Compute a scalar reward by calling an OpenAI-compatible chat API.

    Expects `extra_info` to provide:
      - full_traj: the full trajectory string (required)
      - t: the step index being judged (required)

    The student's two-line output is passed in `solution_str`.
    Returns a dict with "score" plus parsed judge metadata, so reward manager
    can log them while using score for rm_scores.
    """

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for compute_score")

    if not extra_info:
        raise ValueError("extra_info must include full_traj and t")
    full_traj = extra_info.get("full_traj") or extra_info.get("trajectory")
    step_t = extra_info.get("t") or extra_info.get("step") or extra_info.get("step_idx")
    if not full_traj:
        raise ValueError("extra_info.full_traj (or trajectory) is required")
    if step_t is None:
        raise ValueError("extra_info.t (or step/step_idx) is required")

    # Build prompt inline (avoids outdated helper)
    user_prompt = PROMPT_TEMPLATE.format(
        full_traj=full_traj,
        t=step_t,
        student_two_lines=solution_str,
    )

    client = OpenAI(api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=512,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
            # print(resp)
            content = resp.choices[0].message.content.strip()

            parsed: Dict[str, Any]
            score: float
            try:
                parsed = json.loads(content)
                score = float(parsed.get("reward"))
            except Exception:
                # fallback: maybe model returned bare number
                try:
                    score = float(content)
                    parsed = {"gold_label": None, "reward": score, "confidence": None, "brief_feedback": content}
                except Exception as parse_err:  # noqa: BLE001
                    raise ValueError(f"Failed to parse score from response: {content!r}") from parse_err

            # Clip reward to [-1, 1] as per template
            score = max(-1.0, min(1.0, score))
            parsed["reward"] = score
            parsed.setdefault("gold_label", None)
            parsed.setdefault("confidence", None)
            parsed.setdefault("brief_feedback", "")

            return {
                "score": score,
                "gold_label": parsed.get("gold_label"),
                "confidence": parsed.get("confidence"),
                "brief_feedback": parsed.get("brief_feedback"),
                "raw_response": content,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"compute_score failed after {max_retries} retries: {last_err}")


__all__ = ["compute_score"]
