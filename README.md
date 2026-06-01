# onairos-humanapi

Python client for [Onairos Human API](https://human-api.onairos.io), an OpenAI-compatible proxy with user memory and app context. Optional [LangSmith](https://smith.langchain.com) tracing is controlled by environment variables.

This package lives in its own repo (sibling to `OnairosBackEnd`). The backend (`api2.onairos.uk`) handles JWT verification and memory ingestion; **this SDK** talks to the Human API worker.

## Install

When published:

```bash
pip install onairos-humanapi
```

For local development:

```bash
git clone <humanapi-python-repo-url>
cd humanapi
pip install -e ".[dev]"
```

## Credentials

| Variable | Purpose |
|----------|---------|
| `ONAIROS_API_KEY` | **Developer** key from the portal (`ona_…`) — **not** your OpenAI key |
| `ONAIROS_USER_JWT` | End-user JWT from SDK consent (`getAPIurl` / `getAPIurlMobile` `token`) |
| `LANGSMITH_API_KEY` | Optional — tracing only |
| `LANGSMITH_TRACING` | Optional — set to `true` to enable LangSmith traces |
| `LANGSMITH_PROJECT` | Optional — LangSmith project name |

## Quick start

```python
from onairos_humanapi import Onairos

client = Onairos.from_env()

response = client.chat.completions.create(
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

Explicit credentials:

```python
from onairos_humanapi import Onairos

client = Onairos(
    api_key="ona_...",
    user_jwt="<user_jwt_from_handoff>",
)
```

Equivalent raw OpenAI-compatible setup:

```javascript
const client = new OpenAI({
  apiKey: process.env.ONAIROS_API_KEY,
  baseURL: 'https://human-api.onairos.io/v1',
  defaultHeaders: { 'x-jwt-token': onairosUserJWT },
});
```

## LangSmith

Tracing uses a **separate** LangSmith API key. Human API still authenticates with `ONAIROS_API_KEY` + `x-jwt-token`.

No code change is required. Enable tracing with environment variables:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="my-app"
```

Each traced call records clean Human API fields:

```text
memoryInjected
memoryCount
messagesEnriched
userContextInjected
userContextType
provider
model
authMethod
usage
elapsedMs
assistantPreview
```

## Advanced Client

```python
from onairos_humanapi import create_human_api_client, HumanApiConfig

client = create_human_api_client(
    HumanApiConfig(
        api_key="ona_...",
        user_jwt="<user_jwt_from_handoff>",
    )
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Use {onairos_memory}. What should I do next?"}],
)
```

## Tests

```bash
pytest
```

## Project layout

```
Humanapi/
├── src/onairos_humanapi/
│   ├── onairos.py  # primary client interface
│   ├── client.py   # Human API client factory and configuration
│   └── chat.py     # chat completion helpers and tracing
├── tests/
└── examples/
```

## Related

- OnairosBackEnd: `project-memory/HUMAN_API_DEVELOPER_REPORT.md`
- Inference (different product): `apiUrl` + JWT to `api2.onairos.uk` — not Human API worker URL
