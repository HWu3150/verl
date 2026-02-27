"""Smoke test for custom_reward_openai.compute_score.

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/test_custom_reward_openai.py

If no API key is set, the script exits with code 0 after printing a note.
"""

from __future__ import annotations

import os
from pprint import pprint

from omegaconf import OmegaConf

from custom_reward_openai import compute_score


def main():
    cfg = OmegaConf.load("verl/trainer/config/reward/reward.yaml")
    api_key = cfg.custom_reward_function.reward_kwargs.get("api_key")
    base_url = cfg.custom_reward_function.reward_kwargs.get("base_url")
    model = cfg.custom_reward_function.reward_kwargs.get("model") or "gpt-4.1-mini"

    if not api_key:
        print("api_key not set in reward.yaml; skipping live API call.")
        return

    mock_full_traj = (
        "Task: Sort numbers ascending.\n"
        "Step1: Read list [3,1,2].\n"
        "Step2: Compare 3 and 1 -> swap.\n"
        "Step3: Compare 3 and 2 -> swap.\n"
        "Outcome: [1,2,3] sorted."
    )
    mock_student_output = "Reason: swapped larger to the right\nLabel: positive"

    result = compute_score(
        data_source="mock",
        solution_str=mock_student_output,
        extra_info={"full_traj": mock_full_traj, "t": 2},
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    pprint(result)


if __name__ == "__main__":
    main()
