# Koi Configuration Guide

This guide explains all configuration options and common setups.

## Configuration File

Koi stores configuration in `.koi/config.json` in your project directory.

## Configuration Options

| Option | Type | Description | Default | Required |
|--------|------|-------------|---------|----------|
| `api_base` | string | API endpoint URL | - | Yes |
| `api_key` | string | API key (or use env var) | - | Yes* |
| `model` | string | Model identifier | `openai/openai/gpt-5.2-codex` | No |
| `api_format` | string | API format (`responses`, `anthropic`) | Auto-detected | No |
| `max_tokens` | integer | Max tokens per response | 4096 | No |
| `context_window` | integer | Model's context window | 128000 | No |
| `temperature` | float | Sampling temperature (0-2) | Model default | No |
| `skills_paths` | array | Paths to search for skills | `[".koi/skills"]` | No |

*Can use `KOI_API_KEY` environment variable instead

## Common Configurations

### OpenAI-Compatible (Default)

```json
{
  "api_base": "https://api.example.com/v1/responses",
  "api_key": "your-api-key",
  "model": "gpt-4-turbo",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": [".koi/skills"]
}
```

### Anthropic Claude

```json
{
  "api_base": "https://api.anthropic.com/v1/messages",
  "api_key": "sk-ant-...",
  "model": "claude-3-opus-20240229",
  "api_format": "anthropic",
  "max_tokens": 4096,
  "context_window": 200000,
  "skills_paths": [".koi/skills"]
}
```

### Local LLM Server

```json
{
  "api_base": "http://localhost:8080/v1/responses",
  "api_key": "not-needed-for-local",
  "model": "local-model",
  "max_tokens": 2048,
  "context_window": 8192,
  "temperature": 0.7,
  "skills_paths": [".koi/skills", "/shared/skills"]
}
```

## API Key Management

### Environment Variable (Recommended)

```bash
export KOI_API_KEY="your-api-key-here"
koi run
```

### Claude Code Integration

If using Claude Code and authenticated with `claude auth`, Koi automatically uses your API key from `~/.claude.json` when:
- `api_format` is `"anthropic"`
- Model name contains "claude" or "anthropic"
- No API key is provided in config or environment

### Credentials Folder

For multiple credentials, store them in `.koi/credentials/`:

```bash
echo "sk-ant-..." > .koi/credentials/anthropic.key
echo "sk-..." > .koi/credentials/openai.key
```

These are automatically loaded as environment variables when Koi runs commands.

## Model-Specific Settings

### Context Window Guidelines

- **GPT-4 Turbo**: 128,000 tokens
- **Claude 3 Opus**: 200,000 tokens
- **Claude 3 Sonnet**: 200,000 tokens
- **Claude 3 Haiku**: 200,000 tokens
- **GPT-3.5 Turbo**: 16,385 tokens

Set appropriately to avoid context errors.

### Temperature Settings

- `0.0` - Deterministic, best for code/analysis
- `0.7` - Balanced creativity (default)
- `1.0` - Creative responses
- `2.0` - Maximum randomness

## Skills Paths

Configure multiple skill directories:

```json
{
  "skills_paths": [
    ".koi/skills",           // Project-specific skills
    "~/.koi/global-skills",  // Personal skills library
    "/team/shared-skills"    // Team-shared skills
  ]
}
```

Skills are searched in order. First matching skill is used.

## Troubleshooting

### API Format Auto-Detection

Koi auto-detects the API format from the model name:

- Contains "anthropic" or "claude" → `anthropic`
- Otherwise → `responses`

Override by setting `api_format` explicitly.

### Common Issues

1. **"API key not found"**
   - Check `KOI_API_KEY` environment variable
   - Verify `api_key` in config.json
   - For Anthropic, check ~/.claude.json

2. **"Context window exceeded"**
   - Reduce `context_window` setting
   - Use `/compact` command
   - Adjust `max_tokens`

3. **"Model not found"**
   - Verify model name spelling
   - Check provider documentation
   - Ensure `api_format` matches provider

## Advanced Configuration

### Per-Project Overrides

Create project-specific settings:

```bash
cd project1
koi init
# Edit .koi/config.json with project1 settings

cd ../project2
koi init
# Different .koi/config.json for project2
```

### Dynamic Configuration

Load configuration programmatically:

```python
from koi.config import Config

# Load from custom path
config = Config.load(Path("custom/config.json"))

# Create programmatically
config = Config(
    api_base="https://api.anthropic.com",
    api_key=os.getenv("CLAUDE_KEY"),
    model="claude-3-sonnet-20240229"
)
```

## Security Best Practices

1. **Never commit API keys** - Use .gitignore:
   ```
   .koi/config.json
   .koi/credentials/
   ```

2. **Use environment variables** for sensitive data

3. **Rotate keys regularly** and update configs

4. **Limit key permissions** to minimum required

5. **Use project-specific keys** when possible