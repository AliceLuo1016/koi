# Plan: Webhook Server + Slack Integration

## Lessons from OpenClaw

OpenClaw's architecture has several patterns worth adopting:

### 1. Channel abstraction
OpenClaw doesn't hardcode Slack. It has a **channel layer** — each channel (Slack, Telegram, Discord, etc.) is a plugin that normalizes inbound messages into a common format and handles outbound delivery. The agent code never knows which channel it's talking to.

### 2. Dual connection modes
For Slack specifically, OpenClaw supports:
- **Socket Mode** (default) — WebSocket connection initiated by the app. No public URL needed. Simpler to deploy.
- **HTTP Events API** — Slack sends events to a webhook URL. Requires a publicly reachable endpoint.

Both modes normalize into the same internal event format.

### 3. Session routing
Inbound messages are routed to sessions based on context:
- DMs → user session
- Channel messages → channel session
- Thread replies → thread session
These are separate conversation threads with separate context.

### 4. Gateway as the hub
OpenClaw runs a single long-running process (the Gateway) that owns all channel connections, the HTTP server, and the WebSocket control plane. Everything routes through it.

## What Koi Needs

Koi is simpler — it's a team work assistant, not a personal assistant platform. We don't need the full channel abstraction yet. But we do need:

1. **An HTTP server** that can receive webhooks
2. **Slack integration** as the first channel
3. **Session routing** so multiple users can talk to Koi concurrently
4. **A clean boundary** so adding more channels later is straightforward

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Koi Server                      │
│                                                   │
│  ┌─────────┐   ┌──────────┐   ┌───────────────┐ │
│  │  HTTP    │──▶│  Event   │──▶│   Session     │ │
│  │  Server  │   │  Router  │   │   Manager     │ │
│  └─────────┘   └──────────┘   └───────┬───────┘ │
│       ▲                               │          │
│       │              ┌────────────────▼────────┐ │
│  ┌────┴─────┐        │     Agent Instance      │ │
│  │ Channels │        │  (per-session, async)    │ │
│  │  Slack   │        │  ┌─────┐ ┌─────┐ ┌───┐ │ │
│  │  (more)  │        │  │ LLM │ │Tools│ │Mem│ │ │
│  └──────────┘        │  └─────┘ └─────┘ └───┘ │ │
│                      └─────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

### Components

#### 1. HTTP Server (`src/koi/server.py`)
- Lightweight async HTTP server (aiohttp or FastAPI/Starlette)
- Mounts channel webhook routes
- Health check endpoint (`/health`)
- Future: REST API for programmatic access

**Recommendation: use Starlette** (lightweight, async-native, no magic). FastAPI adds OpenAPI docs which is nice but heavier. aiohttp works but less ergonomic. Starlette hits the sweet spot.

#### 2. Channel Interface (`src/koi/channels/base.py`)
Abstract base class that all channels implement:

```python
class Channel(ABC):
    """Base class for messaging channels."""
    
    @abstractmethod
    async def setup(self, app: Starlette) -> None:
        """Register routes on the HTTP server."""
        pass
    
    @abstractmethod
    async def send_message(self, session_id: str, text: str) -> None:
        """Send a message back to the channel."""
        pass
    
    @abstractmethod
    def parse_event(self, raw: dict) -> Optional[InboundMessage]:
        """Normalize a raw webhook event into a common message format."""
        pass
```

Common message format:

```python
@dataclass
class InboundMessage:
    text: str
    user_id: str
    user_name: str
    channel_id: str
    thread_id: Optional[str]
    source: str  # "slack", "webhook", etc.
    raw: dict    # original payload for channel-specific handling
```

#### 3. Slack Channel (`src/koi/channels/slack.py`)

**Phase 1: Socket Mode** (simpler, no public URL needed)
- Uses `slack_sdk.socket_mode.aio.AsyncSocketModeClient`
- Listens for `app_mention` and `message` events
- Sends replies via `slack_sdk.web.aio.AsyncWebClient`
- Handles threading (replies go to the same thread)

**Phase 2: HTTP Events API** (for production/k8s deployment)
- Slack sends events to `/slack/events`
- Verify requests with signing secret
- Respond to URL verification challenges
- Same event handling as Socket Mode after verification

Key Slack behaviors to implement:
- **Ack reaction**: React with 👀 while processing (like OpenClaw)
- **Thread replies**: Always reply in thread to keep channels clean
- **Mention gating**: In channels, only respond when @mentioned
- **DMs**: Respond to all DMs (no mention needed)
- **Typing indicator**: Show "is typing..." while agent is working

#### 4. Session Manager (`src/koi/sessions.py`)

Manages concurrent agent sessions:

```python
class SessionManager:
    """Manages agent sessions for concurrent users."""
    
    async def get_or_create(self, session_key: str, config: Config) -> Agent:
        """Get existing session or create a new one."""
        pass
    
    async def route_message(self, msg: InboundMessage) -> str:
        """Route an inbound message to the right session and return the response."""
        pass
    
    async def cleanup_idle(self, max_idle_seconds: int) -> None:
        """Clean up sessions that have been idle too long."""
        pass
```

Session key format:
- DM: `slack:dm:<user_id>`
- Channel: `slack:channel:<channel_id>`
- Thread: `slack:channel:<channel_id>:thread:<thread_ts>`

Each session gets its own Agent instance with its own conversation history and memory.

#### 5. Configuration

Extend `koi.config.json`:

```json
{
  "server": {
    "enabled": false,
    "host": "0.0.0.0",
    "port": 8080
  },
  "channels": {
    "slack": {
      "enabled": false,
      "mode": "socket",
      "app_token": "xapp-...",
      "bot_token": "xoxb-...",
      "signing_secret": "...",
      "mention_only_in_channels": true,
      "ack_reaction": "eyes",
      "max_idle_session_minutes": 60
    }
  }
}
```

## Implementation Plan

### Step 1: HTTP Server + Health Check
- Add `starlette` dependency
- Create `src/koi/server.py` with basic app, `/health` endpoint
- Add `koi serve` CLI command that starts the server
- Server runs the async event loop; agent sessions are async tasks within it

### Step 2: Channel Abstraction
- Create `src/koi/channels/` package
- Define `Channel` base class and `InboundMessage` datatype
- Define `OutboundMessage` for replies (text, thread_id, reactions)

### Step 3: Session Manager
- Create `src/koi/sessions.py`
- Session creation, routing, idle cleanup
- Each session wraps an `Agent` instance
- Shared config, separate conversation state

### Step 4: Slack Channel (Socket Mode)
- Add `slack-sdk` dependency (with `[socket-mode]` extra)
- Implement `SlackChannel` with Socket Mode connection
- Handle: app_mention, message.im, message.channels
- Reply in threads, ack with reaction, typing indicator
- Wire into session manager for message routing

### Step 5: CLI Integration
- `koi serve` — start server with all enabled channels
- `koi serve --channel slack` — start with specific channel only
- Config validation on startup (check tokens are set)
- Graceful shutdown (close channels, save sessions)

### Step 6: Slack HTTP Mode
- Add `/slack/events` webhook route
- Request signing verification
- URL verification challenge handling
- Same event processing as Socket Mode

## Dependencies to Add

```toml
[project.optional-dependencies]
server = [
    "starlette>=0.30.0",
    "uvicorn>=0.23.0",
    "slack-sdk[socket-mode]>=3.21.0",
]
```

Install with: `pip install 'koi[server]'` or `pip install '.[server]'`

## File Structure

```
src/koi/
├── server.py              # HTTP server (Starlette app)
├── sessions.py            # Session manager
├── channels/
│   ├── __init__.py
│   ├── base.py            # Channel ABC + InboundMessage
│   └── slack.py           # Slack channel (Socket + HTTP modes)
```

## Key Design Decisions

1. **Socket Mode first** — No public URL needed. Developers can run `koi serve` on their laptop and it just works. HTTP mode is for production.

2. **Starlette over FastAPI** — Lighter weight. We don't need OpenAPI docs for internal webhooks. Can always upgrade later.

3. **One Agent per session** — Each user/channel/thread gets its own Agent with its own context. No shared conversation state (but shared config and skills).

4. **Optional dependency** — Server mode is `pip install 'koi[server]'`. The CLI-only experience stays lightweight.

5. **Channel abstraction from day one** — Even though Slack is first, the abstraction is cheap and saves massive refactoring later when adding Teams, Discord, etc.

## Not in Scope (Yet)

- Multi-agent within a session (sub-agents already exist)
- Persistent session storage (in-memory for now, disk persistence later)
- Authentication for the HTTP API itself (internal network assumed)
- Rate limiting (Slack SDK handles Slack rate limits)
- Message queuing (direct processing for now)
