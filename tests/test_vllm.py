from __future__ import annotations

import json
import re
from openai import OpenAI


def main() -> None:
    client = OpenAI(
        api_key="dummy",
        base_url="http://127.0.0.1:8003/v1",
    )

    user_prompt = """
Return a valid JSON object with this schema:
{
  "message": string,
  "contains_vllm": boolean
}

The message should say hello in one sentence and include the word "vLLM".
""".strip()

    resp = client.chat.completions.create(
        model="qwen3-32b",
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=512,
        temperature=0.2,
    )

    msg = resp.choices[0].message

    # thinking 部分
    print("Reasoning:\n", getattr(msg, "reasoning_content", None))

    # 最终回答部分（这里应该是 JSON）
    print("Content:\n", msg.content)

    # 只解析 content，不解析 reasoning_content
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    data = json.loads(msg.content)
    print("Parsed JSON:\n", data)


if __name__ == "__main__":
    main()