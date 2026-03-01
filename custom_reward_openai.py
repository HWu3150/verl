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
import re
import time
from typing import Any, Dict

from openai import OpenAI

PROMPT_TEMPLATE = """You are an expert hindsight judge for action-observation trajectories.

You will be given:
- FULL trajectory: initial observation, task, steps 1..T, and outcome.
- Student output for step t: (ignore the thinking part) two lines:
    Reason: ...
    Label: negative|neutral|positive

Your job:
1) Decide whether the student's LABEL is correct in hindsight (use the full trajectory, events after step t allowed).
2) Produce a scalar reward in [-1, 1]:
   - exact label match → about +0.8
   - adjacent mismatch (positive↔neutral or neutral↔negative) → about +0.2
   - opposite mismatch (positive↔negative) → about -0.8
   - invalid format/label → -1.0
   - if the Reason contradicts facts, subtract ~0.2; if it clearly supports, add ~0.2; generic → 0.
   - clip to [-1, 1].
3) Do NOT reward verbosity; ambiguity → prefer neutral and keep reward near 0.

Return ONLY the reward number (float). No JSON, no explanations.

FULL trajectory (steps 1..T):
{full_traj}

We are judging step t={t}.
Student output (two lines):
{student_two_lines}

Now output the reward number only:
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
    enable_thinking: bool | None = None,
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

    def _strip_wrappers(text: str) -> str:
        """Remove <think>...</think> prefix and code fences like ```json ... ```."""
        s = text.strip()
        if "<think>" in s and "</think>" in s:
            s = s.split("</think>", 1)[1]
        if s.startswith("```"):
            # drop leading fence with optional language
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
            # drop trailing fence
            s = re.sub(r"\n?```$", "", s)
        return s.strip()

    last_err = None
    for attempt in range(max_retries):
        try:
            extra_body = None
            if enable_thinking is not None:
                extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=2048,
                timeout=timeout,
                extra_body=extra_body,
            )
            raw_content = resp.choices[0].message.content or resp.choices[0].message.reasoning_content or ""
            if isinstance(raw_content, list):
                raw_content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_content)
            raw_content = str(raw_content)

            # Print full raw answer for debugging (includes think + fences if any)
            print("[reward debug] raw_response:", raw_content)

            content = _strip_wrappers(raw_content)

            score: float
            try:
                score = float(content)
            except Exception:  # noqa: BLE001
                print("Failed to parse score from response")
                score = 0.0

            # Clip reward to [-1, 1] as per template
            score = max(-1.0, min(1.0, score))
            return {"score": score, "raw_response": content}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"compute_score failed after {max_retries} retries: {last_err}")


__all__ = ["compute_score"]
