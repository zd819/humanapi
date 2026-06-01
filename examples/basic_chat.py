"""Example: Human API chat completion with optional LangSmith tracing.

Environment:
  ONAIROS_API_KEY     - developer key from the Onairos portal
  ONAIROS_USER_JWT    - user JWT from SDK consent handoff
  LANGSMITH_TRACING   - set to true to enable traces
  LANGSMITH_API_KEY   - LangSmith API key, required only when tracing
  LANGSMITH_PROJECT   - LangSmith project name, optional
"""

from onairos_humanapi import Onairos


def main() -> None:
    try:
        client = Onairos.from_env()
    except ValueError as exc:
        print(exc)
        return

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Use {onairos_memory} and the current app context. "
                    "What should I work out today?"
                ),
            }
        ],
        user_context={
            "currentPage": "workout_builder",
            "currentGoal": "improve cardio endurance",
            "availableTimeMinutes": 25,
        },
        run_name="workout_recommendation",
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
