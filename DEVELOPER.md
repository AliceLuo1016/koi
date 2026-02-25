# Koi Developer Guide

This guide explains how to extend and customize Koi for your needs.

## Architecture Overview

Koi follows a modular architecture with clear separation of concerns:

```
User Input → CLI → Agent → LLM Client → Tools/Skills → Output
                     ↓
                  Memory
```

## Extending Koi

### Adding Custom Tools

Tools are the primary way Koi interacts with the world. To add a new tool:

1. **Define the tool in `tools.py`:**

```python
def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        # ... existing tools ...
        {
            "type": "function",
            "function": {
                "name": "your_tool_name",
                "description": "What your tool does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter description"},
                        "param2": {"type": "integer", "description": "Optional param"},
                    },
                    "required": ["param1"]
                }
            }
        }
    ]
```

2. **Implement the tool in `ToolExecutor.execute()`:**

```python
async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
    # ... existing tools ...
    
    elif tool_name == "your_tool_name":
        return await self._your_tool_implementation(
            arguments["param1"],
            arguments.get("param2", default_value)
        )
```

3. **Add the implementation method:**

```python
async def _your_tool_implementation(self, param1: str, param2: int = 0) -> str:
    """Your tool logic here."""
    # Implement your functionality
    # Use self.sandbox for security checks if needed
    return "Tool result"
```

### Creating Custom Skills

Skills are markdown files that teach Koi how to perform complex tasks. Use the skill-creator:

```bash
koi run
> Use the skill-creator to make a new skill called "database-backup" that backs up PostgreSQL databases
```

Or manually create a skill:

1. Create directory: `.koi/skills/your-skill/`
2. Add `SKILL.md` with this structure:

```markdown
# Your Skill Name

Brief description of what this skill does.

## When to Use

- Specific trigger condition 1
- Specific trigger condition 2

## Process

1. First step with specific command
2. Second step with error handling
3. Final verification step

## Example

Show a concrete example of the skill in action.
```

### Customizing the LLM Client

To add support for a new LLM provider:

1. **Update `Config` class in `config.py`:**

```python
def __init__(self, ...):
    # Add auto-detection logic
    if "your-provider" in model:
        self.api_format = "your-provider"
```

2. **Extend `LLMClient` in `llm.py`:**

```python
def __init__(self, config: Config):
    # ... existing code ...
    if config.api_format == "your-provider":
        self.headers = {
            # Your provider's required headers
        }

async def chat(self, messages: List[Dict[str, Any]], ...):
    if self.config.api_format == "your-provider":
        # Convert messages to your provider's format
        # Make API call
        # Convert response back
```

### Working with the Sandbox

The sandbox protects users from accidental damage. To modify sandbox rules:

1. Edit `.koi/sandbox.yaml` in your project
2. Test thoroughly with `sandbox.check_*` methods
3. Consider security implications

Example custom sandbox check:

```python
def check_custom_operation(self, operation: str) -> Tuple[bool, Optional[str]]:
    """Check if a custom operation is allowed."""
    if "dangerous_pattern" in operation:
        return False, "Operation blocked: contains dangerous pattern"
    return True, None
```

### Testing Your Extensions

1. **Unit tests** in `tests/test_your_feature.py`:

```python
import pytest
from koi.your_module import YourClass

async def test_your_feature():
    instance = YourClass()
    result = await instance.your_method()
    assert result == expected_value
```

2. **Integration tests** with the full agent:

```python
async def test_agent_with_your_tool(mock_llm):
    agent = Agent(config)
    # Mock LLM to return tool call
    # Verify tool execution
```

3. **Run tests:**

```bash
pytest tests/test_your_feature.py -v
```

## Best Practices

1. **Security First**: Always validate inputs and use the sandbox
2. **Async by Default**: Use async/await for I/O operations
3. **Clear Error Messages**: Help users understand what went wrong
4. **Document Everything**: Update skills, tools, and code comments
5. **Test Thoroughly**: Unit tests, integration tests, and manual testing

## Common Patterns

### Progress Indication

For long-running operations:

```python
from rich.progress import Progress

async def long_operation():
    with Progress() as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            # Do work
            progress.update(task, advance=1)
```

### Error Handling

Always provide helpful error messages:

```python
try:
    result = await risky_operation()
except SpecificError as e:
    return f"Operation failed: {e}. Try doing X instead."
except Exception as e:
    return f"Unexpected error: {e}. Please report this issue."
```

### Memory Updates

When learning something new:

```python
if learn_from_this:
    self.memory.append(
        "## Learned Pattern\n"
        f"When {condition}, use {solution} instead of {old_approach}"
    )
```

## Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use rich's inspect for complex objects:

```python
from rich import inspect
inspect(complex_object, methods=True)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `black` and `ruff` for formatting
5. Submit a pull request

For questions, create an issue on GitLab.