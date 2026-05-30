"""
Example: Human API chat with optional LangSmith tracing.

Environment:
  ONAIROS_API_KEY     — developer key (ona_…)
  ONAIROS_USER_JWT    — user JWT from SDK consent handoff
  LANGSMITH_TRACING   — set to true to enable traces (optional)
  LANGSMITH_API_KEY   — LangSmith project key (optional)
"""

import os

from onairos_humanapi import (
    build_chat_request_kwargs,
    chat_completion_traced,
    create_traced_human_api_client,
    from_env,
)


def main() -> None:
    user_jwt = os.environ.get("ONAIROS_USER_JWT", "")
    api_key = os.environ.get("ONAIROS_API_KEY", "")

    if not api_key or not user_jwt:
        print("Set ONAIROS_API_KEY and ONAIROS_USER_JWT to run this example.")
        return

    messages = [
        {
            "role": "user",
            "content": (
                "Use {onairos_memory} and the current app context. "
                "What should I work out today?"
            ),
        }
    ]
    user_context = {
        "currentPage": "workout_builder",
        "currentGoal": "improve cardio endurance",
        "availableTimeMinutes": 25,
    }

    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")

    if tracing:
        response = chat_completion_traced(
            api_key=api_key,
            user_jwt=user_jwt,
            messages=messages,
            user_context=user_context,
            run_name="workout_recommendation",
        )
    else:
        client = from_env()
        kwargs = build_chat_request_kwargs(
            messages=messages,
            user_context=user_context,
        )
        response = client.chat.completions.create(**kwargs)

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
