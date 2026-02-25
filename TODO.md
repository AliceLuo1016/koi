# Koi TODO List

## High Priority

### 1. Implement `web_search` Tool

**Status**: Placeholder returns TODO message

**Location**: `src/koi/tools.py` - `_web_search()` method

**Requirements**:
- Integrate with Brave Search API or similar
- Parse search results into useful format
- Handle rate limiting and errors
- Return structured results

**Proposed Implementation**:
```python
async def _web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search the web using Brave Search API."""
    api_key = self.sandbox.get_credential("brave_search")
    if not api_key:
        return {"error": "No Brave Search API key found in .koi/credentials/brave_search.key"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key},
            params={"q": query, "count": num_results}
        )
        
    # Parse and return results
    return {
        "query": query,
        "results": [...],
        "timestamp": datetime.now().isoformat()
    }
```

## Medium Priority

### 2. Skills Marketplace

- Central repository for sharing skills
- Version management for skills
- Dependency handling
- Rating/review system

### 3. Plugin System

- Allow third-party tool additions without modifying core
- Dynamic tool loading
- Tool namespacing to prevent conflicts

### 4. Enhanced Context Management

- Smarter compaction strategies
- Preserve important context better
- User-defined compaction rules

### 5. Telemetry (Opt-in)

- Usage analytics to improve koi
- Error reporting
- Performance metrics
- Fully opt-in with clear privacy

## Low Priority

### 6. GUI Frontend

- Web-based interface option
- Electron app for desktop
- Mobile companion app

### 7. Voice Interface

- Speech-to-text input
- Text-to-speech output
- Wake word detection

### 8. Multi-Agent Coordination

- Allow multiple koi instances to collaborate
- Shared memory between agents
- Task delegation

## Completed

- ✅ Anthropic Claude support
- ✅ Markdown rendering fix
- ✅ Ctrl+C interrupt handling
- ✅ Sandbox security system
- ✅ Cron integration
- ✅ Alerts system
- ✅ Skills framework

## Contributing

See DEVELOPER.md for implementation guidelines. Priority items welcome for contribution!