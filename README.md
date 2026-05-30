# onairos-humanapi

Python client for [Onairos Human API](https://human-api.onairos.io) — an OpenAI-compatible proxy with user memory and app context. Optional [LangSmith](https://smith.langchain.com) tracing via `langsmith.wrappers.wrap_openai`.

This package lives in its own repo (sibling to `OnairosBackEnd`). The backend (`api2.onairos.uk`) handles JWT verification and memory ingestion; **this SDK** talks to the Human API worker.

## Install

```bash
cd Humanapi
pip install -e ".[dev]"
```

## Credentials

| Variable | Purpose |
|----------|---------|
| `ONAIROS_API_KEY` | **Developer** key from the portal (`ona_…`) — **not** your OpenAI key |
| `ONAIROS_USER_JWT` | End-user JWT from SDK consent (`getAPIurl` / `getAPIurlMobile` `token`) |
| `LANGSMITH_API_KEY` | Optional — tracing only |
| `LANGSMITH_TRACING` | `true` to wrap the client with LangSmith |

## Quick start

```python
from onairos_humanapi import create_traced_human_api_client, chat_completion, HumanApiConfig

config = HumanApiConfig(
    api_key="ona_…",
    user_jwt="<user_jwt_from_handoff>",
)

client = create_traced_human_api_client(config, enable_tracing=True)

response = chat_completion(
    client,
    messages=[{
        "role": "user",
        "content": "Use {onairos_memory} and the current app context. What should I work out today?",
    }],
    model="openai/gpt-4o-mini",
    user_context={
        "currentPage": "workout_builder",
        "currentGoal": "improve cardio endurance",
        "availableTimeMinutes": 25,
    },
)

print(response.choices[0].message.content)
```

Equivalent to the JS pattern:

```javascript
const client = new OpenAI({
  apiKey: process.env.ONAIROS_API_KEY,
  baseURL: 'https://human-api.onairos.io/v1',
  defaultHeaders: { 'x-jwt-token': onairosUserJWT },
});
```

## LangSmith

Tracing uses a **separate** LangSmith API key. Human API still authenticates with `ONAIROS_API_KEY` + `x-jwt-token`.

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_…"
os.environ["LANGSMITH_PROJECT"] = "my-app"  # optional

from onairos_humanapi import from_env

client = from_env(traced=True)
```

Or wrap manually:

```python
from openai import OpenAI
from langsmith import wrappers
from onairos_humanapi import create_human_api_client, HumanApiConfig

raw = create_human_api_client(HumanApiConfig(api_key="ona_…", user_jwt="…"))
client = wrappers.wrap_openai(raw)
```

## Tests

```bash
pytest
```

## Project layout

```
Humanapi/
├── src/onairos_humanapi/
│   ├── client.py   # OpenAI client + LangSmith wrap
│   └── chat.py     # chat_completion helpers
├── tests/
└── examples/
```

## Related

- OnairosBackEnd: `project-memory/HUMAN_API_DEVELOPER_REPORT.md`
- Inference (different product): `apiUrl` + JWT to `api2.onairos.uk` — not Human API worker URL
