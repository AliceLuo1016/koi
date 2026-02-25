# Koi Troubleshooting Guide

Common issues and solutions for Koi users.

## Installation Issues

### "koi: command not found"

**Problem**: After installation, `koi` command isn't recognized.

**Solutions**:
```bash
# Check if koi is in PATH
which koi

# If using pip install -e (development mode)
pip show koi

# Reinstall
cd ~/koi
pip install -e .

# Or for production
pip install .
```

### Import errors

**Problem**: `ModuleNotFoundError` when running koi.

**Solution**:
```bash
# Ensure all dependencies are installed
pip install -e ".[dev]"  # For development
# or
pip install .  # For production
```

## API Connection Issues

### "API key not found"

**Problem**: Koi can't find your API key.

**Solutions**:

1. Set environment variable:
   ```bash
   export KOI_API_KEY="your-key-here"
   ```

2. Add to config.json:
   ```json
   {
     "api_key": "your-key-here"
   }
   ```

3. For Anthropic + Claude Code:
   ```bash
   claude auth  # Authenticate Claude Code first
   ```

### "Connection refused" or timeout errors

**Problem**: Can't connect to API endpoint.

**Solutions**:

1. Check API endpoint URL:
   ```bash
   curl -X POST https://your-api-endpoint/test
   ```

2. Verify network connectivity:
   ```bash
   ping api.anthropic.com
   ```

3. Check proxy settings:
   ```bash
   # If behind proxy
   export HTTP_PROXY="http://proxy:port"
   export HTTPS_PROXY="http://proxy:port"
   ```

### "Model not found" errors

**Problem**: The specified model isn't recognized.

**Solutions**:

1. Check model name spelling:
   ```json
   {
     "model": "claude-3-opus-20240229"  // Exact name
   }
   ```

2. Verify API format:
   ```json
   {
     "api_format": "anthropic"  // or "responses"
   }
   ```

## Runtime Issues

### "Context window exceeded"

**Problem**: Conversation too long for model's context.

**Solutions**:

1. Force compaction:
   ```
   koi> /compact
   ```

2. Reduce context window in config:
   ```json
   {
     "context_window": 100000  // Lower value
   }
   ```

3. Start fresh session:
   ```
   koi> /exit
   $ koi run
   ```

### Ctrl+C not working

**Problem**: Can't interrupt long-running operations.

**Solution**: This was fixed in recent versions. Update koi:
```bash
cd ~/koi
git pull
pip install -e .
```

### Markdown not rendering properly

**Problem**: Backticks and bold text show as plain text.

**Solution**: Fixed in latest version. Update koi or check terminal support:
```bash
# Test terminal capabilities
echo -e "\033[1mBold\033[0m \033[3mItalic\033[0m"
```

## Sandbox Security Issues

### "Access denied" for file operations

**Problem**: Can't read/write certain files.

**Solutions**:

1. Check sandbox.yaml:
   ```yaml
   filesystem:
     allowed_paths:
       - "."
       - "/specific/path"  # Add your path
   ```

2. Use relative paths:
   ```
   koi> Read ./file.txt  # Good
   koi> Read /etc/passwd  # Blocked
   ```

### "Command blocked"

**Problem**: Shell command rejected by sandbox.

**Solutions**:

1. Check blocked patterns in sandbox.yaml
2. Use safer alternatives
3. Add to confirm_patterns for user approval:
   ```yaml
   commands:
     confirm_patterns:
       - "your_command"
   ```

### Missing environment variables

**Problem**: Commands fail due to missing env vars.

**Solution**: Add to sandbox.yaml allowlist:
```yaml
environment:
  allowlist:
    - PATH
    - HOME
    - YOUR_VARIABLE  # Add here
```

## Cron Issues

### Cron jobs not running

**Problem**: Scheduled tasks don't execute.

**Solutions**:

1. Check crontab:
   ```bash
   crontab -l | grep koi
   ```

2. Verify koi path:
   ```bash
   which koi  # Should return full path
   ```

3. Check logs:
   ```bash
   ls -la .koi/cron-logs/
   tail -f .koi/cron-logs/latest.log
   ```

4. Test manually:
   ```bash
   koi run --task "Your task here" --non-interactive
   ```

### "koi: command not found" in cron

**Problem**: Cron can't find koi binary.

**Solution**: Recent versions handle this automatically. Update koi or check:
```bash
# In .koi/cron-scripts/*.sh files
cat .koi/cron-scripts/*.sh | grep PATH
```

## Skills Issues

### Skill not found

**Problem**: Koi doesn't recognize your skill.

**Solutions**:

1. Check skill location:
   ```bash
   find .koi/skills -name "SKILL.md"
   ```

2. Verify skill format:
   ```bash
   # Must have proper markdown structure
   cat .koi/skills/your-skill/SKILL.md
   ```

3. List available skills:
   ```
   koi> /skills
   ```

### Skill not triggering

**Problem**: Skill exists but doesn't activate.

**Solutions**:

1. Check skill matching in SKILL.md:
   ```markdown
   ## When to Use
   - Specific trigger phrases
   - Clear conditions
   ```

2. Use exact skill name:
   ```
   koi> Use the log-monitor skill
   ```

## Memory Issues

### Memory not persisting

**Problem**: Koi forgets things between sessions.

**Solutions**:

1. Check memory file:
   ```bash
   cat .koi/MEMORY.md
   ```

2. Use /remember explicitly:
   ```
   koi> /remember Important fact here
   ```

3. Ensure write permissions:
   ```bash
   ls -la .koi/MEMORY.md
   chmod 644 .koi/MEMORY.md
   ```

## Performance Issues

### Slow responses

**Problem**: Koi takes too long to respond.

**Solutions**:

1. Check API latency:
   ```bash
   time curl -X POST your-api-endpoint
   ```

2. Reduce max_tokens:
   ```json
   {
     "max_tokens": 2048  // Lower value
   }
   ```

3. Use faster model:
   ```json
   {
     "model": "claude-3-haiku-20240307"  // Faster
   }
   ```

## Getting Help

### Debug mode

Enable verbose logging:
```python
# In your code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Report issues

1. Check existing issues on GitLab
2. Include:
   - Koi version (`pip show koi`)
   - Error message
   - Steps to reproduce
   - Config (without API key)

### Community support

- GitLab Issues: Report bugs
- Discussions: Ask questions
- Contributing: See DEVELOPER.md