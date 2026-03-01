# TODO

## bug fix: after sub agent notify completion, the line koi>  for user input is not displayed.
  ✓ Sub-agent Repo Sync (id=11839b90) completed
    ✓ Repository sync completed successfully!
    Summary:
    - Synced files from GitHub (/home/aliluo/git/github/koi) to GitLab (/home/aliluo/git/koi)
    - Updated: `src/koi/agent.py`
    - Committed with message: "Sync from GitHub: Fix: Show full multi-line sub-agent results instead of truncated single line (github:9167f3c)"
    - Pushed to GitLab main branch
    The sync preserved the GitLab-specific directories (.koi/, src/koi/bundled_skills/) as configured in the skill.
(there is no koi> after the sub agent notify completion)

## in the final token usage summary. could you show the cache hit ratio?
koi> /exit                                                                                                                                         
👋 Goodbye!

Token Usage:
  Input:  662 tokens
  Output: 266 tokens
  Total:  928 tokens
  Requests: 4
  Cache read:     8,310 tokens
  Cache creation: 8,860 tokens
  Est. cost: $0.2085

## when use /usage, show the usage of the current session and the past 7 days usage

## could you check if we are accidentally submitting credentials in github?

## improve test coverage 

## check code from openclaw and cross compare the code in koi and openclaw. update this todo list with things that you think are important to add to koi.

## check code from codex and cross compare the code in koi and codex. update this todo list with things that you think are important to add to koi.
## Cross-comparison findings

Based on analysis of openclaw (TypeScript) and codex (Rust) architectures, the following improvements are recommended for koi, ordered by priority/impact:

### (P1) Enhanced Error Handling and Recovery
- Implement sophisticated error classification system (billing errors, auth errors, rate limits, context overflow)
- Add automatic retry logic with exponential backoff for transient failures
- Implement auth profile rotation and failover mechanisms for API key management
- Add graceful degradation for different error types instead of generic failures

### (P2) Advanced Context Window Management
- Implement proactive context pruning with smart summarization before hitting limits
- Add context budget enforcement with configurable thresholds and warnings
- Implement tool result truncation for oversized outputs while preserving essential information
- Add context window utilization monitoring and alerts

### (P3) Tool Execution Safety and Sandboxing
- Add comprehensive tool execution policies and allowlist management
- Implement tool loop detection to prevent recursive or infinite tool calls
- Add tool execution approval system for sensitive operations
- Implement sandbox environment validation and security checks

### (P4) Robust Streaming and Real-time Features
- Improve streaming response handling with better chunk management and error recovery
- Add streaming state management to handle interruptions and reconnections
- Implement proper cleanup for cancelled or interrupted streams
- Add streaming performance metrics and monitoring

### (P5) Session and Subagent Management
- Implement session lifecycle events and proper cleanup hooks
- Add subagent depth limiting and resource management
- Implement session persistence and state recovery
- Add inter-agent communication and coordination mechanisms

### (P6) Usage Tracking and Analytics
- Add detailed usage normalization and cost calculation per provider
- Implement usage accumulation across multiple API calls in a session
- Add usage pattern analysis and reporting
- Implement cache hit ratio tracking and optimization recommendations

### (P7) Configuration and Model Management
- Add model discovery and fallback configuration
- Implement provider-specific authentication and key rotation
- Add model compatibility checking and automatic provider selection
- Implement configuration validation and environment-specific overrides

### (P8) Memory and Skills Architecture
- Implement workspace-aware skill loading and management
- Add skill dependency resolution and conflict detection
- Implement memory search and retrieval mechanisms
- Add skill versioning and update management

### (P9) Security and Compliance
- Add comprehensive credential scrubbing and secret detection
- Implement audit logging for all tool executions and sensitive operations
- Add user permission validation and access control
- Implement secure token handling and storage mechanisms

### (P10) Development and Testing Infrastructure
- Add comprehensive integration testing framework
- Implement mock providers for testing without API costs
- Add performance benchmarking and regression testing
- Implement development mode with enhanced debugging and introspection
