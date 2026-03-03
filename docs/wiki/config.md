# Configuration Reference

All configuration for Koi lives in `.koi/config.json` in the project root. Configuration can also be set via environment variables and CLI flags.

## Config File Location

```
<project-root>/
└── .koi/
    └── config.json
```

Loaded by `Config.load()` — defaults to `Path.cwd() / ".koi" / "config.json"`.

## All Config Fields

```json
{
  "api_base": "https://api.example.com/v1/responses",
  "api_key": "sk-...",
  "model": "openai/openai/gpt-5.2-codex",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": [".koi/skills"],
  "temperature": null,
  "api_format": "responses",
  "thinking_level": "low",
  "prompt_caching": true,
  "debug": false,
  "server": {
    "enabled": false,
    "host": "0.0.0.0",
    "port": 8080
  },
  "channels": { }
}
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_base` | string | `""` | Base URL for the LLM API endpoint |
| `api_key` | string | `""` | API key (see [API Key Resolution](#api-key-resolution)) |
| `model` | string | `"openai/openai/gpt-5.2-codex"` | Model identifier sent to the API |
| `max_tokens` | int | `4096` | Maximum output tokens per LLM call |
| `context_window` | int | `128000` | Context window size in tokens (used for all context management layers) |
| `skills_paths` | list[str] | `[".koi/skills"]` | Directories to scan for `SKILL.md` files |
| `temperature` | float\|null | `null` | Sampling temperature. `null` uses provider default. Omitted when thinking is enabled (Anthropic requirement). |
| `api_format` | string\|null | auto-detected | API format: `"anthropic"`, `"chat_completions"`, or `"responses"` |
| `thinking_level` | string | `"low"` | Extended thinking level: `"off"`, `"minimal"`, `"low"`, `"medium"`, `"high"` |
| `prompt_caching` | bool | `true` | Enable Anthropic prompt caching (only applies to `api_format=anthropic`) |
| `debug` | bool | `false` | Enable debug transcript logging to `.koi/transcript.jsonl` |
| `server` | object | — | Server config for `koi serve` |
| `server.enabled` | bool | `false` | Enable the HTTP server |
| `server.host` | string | `"0.0.0.0"` | Server bind address |
| `server.port` | int | `8080` | Server port |
| `channels` | object | `{}` | Channel configurations (e.g., Slack) |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `KOI_API_KEY` | API key override (takes priority over config file) |
| `KOI_SPAWN_DEPTH` | Current sub-agent nesting depth (set automatically by SubagentManager, not user-set) |

## API Key Resolution

API keys are resolved in priority order (`config.py:103`):

```
1. Explicit api_key in config.json
2. KOI_API_KEY environment variable
3. Claude Code config (~/.claude.json → primaryApiKey)  [anthropic format only]
```

The Claude Code fallback (`load_claude_code_api_key()`) reads `~/.claude.json` and extracts the `primaryApiKey` field, but only if it starts with `sk-ant-`.

## API Format Auto-Detection

When `api_format` is not explicitly set in config, Koi auto-detects from the model name (`config.py:83`):

```python
if "anthropic" in model or "claude" in model:
    self.api_format = "anthropic"
else:
    self.api_format = "responses"
```

| Model name contains | Auto-detected format |
|-----|------|
| `anthropic` or `claude` | `anthropic` |
| Anything else | `responses` |

To use Chat Completions, set `"api_format": "chat_completions"` explicitly.

## Thinking Level Normalization

`normalize_think_level()` (`config.py:12`) maps user-friendly aliases to canonical levels:

```python
THINK_LEVELS = ("off", "minimal", "low", "medium", "high")
```

| User Input | Normalized To |
|------------|---------------|
| `off`, `disabled`, `none` | `off` |
| `on`, `enable`, `enabled` | `low` |
| `min`, `minimal`, `think` | `minimal` |
| `low` | `low` |
| `med`, `mid`, `medium` | `medium` |
| `high`, `max`, `ultra` | `high` |

### Per-Provider Thinking Implementation

The thinking level is translated differently per API format (see [LLM Client](llm-client.md)):

| Format | Parameter | Mapping |
|--------|-----------|---------|
| Anthropic | `thinking.budget_tokens` | minimal→1024, low→2048, medium→8192, high→16384 |
| Chat Completions | `reasoning_effort` | minimal→low, low→medium, medium→medium, high→high |
| Responses | `reasoning.effort` | Same as Chat Completions |

If the model doesn't support thinking (`supports_thinking()` returns `False`), Koi falls back to prompt-based `<think>/<final>` tags via `use_reasoning_tags`.

## `effective_thinking_level()`

The `Config` class has a method that checks whether the model actually supports thinking:

```python
def effective_thinking_level(self):
    if self.thinking_level == "off":
        return "off"
    if supports_thinking(self.model, self.api_format):
        return self.thinking_level
    return "off"
```

## `spawn_depth` Property

```python
@property
def spawn_depth(self) -> int:
    return int(os.environ.get("KOI_SPAWN_DEPTH", "0"))
```

Reads the current sub-agent nesting depth from the environment. Set by `SubagentManager` when spawning children — the parent increments `KOI_SPAWN_DEPTH` for the child process.

## CLI Model Presets

`koi init` offers model presets (`cli.py:31`):

| # | Name | Format | Context |
|---|------|--------|---------|
| 1 | GPT-5.2 Codex | `responses` | 128K |
| 2 | Claude Opus 4.6 (via NVIDIA) | `chat_completions` | 200K |
| 3 | Claude Opus 4.6 (via Claude Code) | `anthropic` | 200K |

## Creating Default Config

`create_default_config()` generates a minimal config dict:

```python
def create_default_config(
    model="openai/openai/gpt-5.2-codex",
    api_base="",
    api_key="",
    api_format="responses",
    context_window=128000,
):
    return {
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "max_tokens": 4096,
        "context_window": context_window,
        "skills_paths": [".koi/skills"],
        "api_format": api_format,
    }
```

## Related Pages

- [LLM Client](llm-client.md) — How config affects API calls, thinking, caching
- [Sandbox Security](sandbox.md) — `sandbox.yaml` configuration
- [Skills System](skills.md) — `skills_paths` configuration
