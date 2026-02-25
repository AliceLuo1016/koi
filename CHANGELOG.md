# Changelog

All notable changes to Koi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Anthropic Claude API support with auto-detection from model names
- Integration with Claude Code configuration (`~/.claude.json`)
- Auto-detection of API format based on model name
- Desktop notifications for alerts (macOS/Linux)
- Structured alerts system with severity levels
- Sandbox security system with three-layer protection
- Cron job management with natural language tasks
- Multi-line input support with Escape+Enter
- Context compaction for long conversations
- Rich markdown rendering in terminal output
- Bundled skills for common workflows (Lepton, curation stats)
- Skill creator tool for building custom skills
- Memory separation between general and skill-specific learnings

### Changed
- Improved markdown rendering using `rich.markdown.Markdown` instead of `rich.text.Text`
- Enhanced Ctrl+C handling for instant interruption of operations
- Better signal handling in async operations
- Increased retry backoff for API calls
- LLM thinking text now displays before tool execution

### Fixed
- Markdown formatting (backticks and bold text) now renders correctly
- Ctrl+C signal handling improved for immediate response
- Tool execution output formatting

### Security
- Added comprehensive sandbox system for file access control
- Environment variable scrubbing with allowlist
- Command pattern blocking and confirmation prompts
- Credentials management system in `.koi/credentials/`

## [0.1.0] - 2024-02-16

### Added
- Initial release of Koi
- Core agent conversation loop
- Tool system with file operations, command execution, and web tools
- Persistent memory system
- Skills framework with markdown-based definitions
- System crontab integration
- Configuration management
- OpenAI-compatible API support

### Technical Details
- Built with Python 3.9+
- Uses prompt_toolkit for enhanced terminal input
- Rich library for beautiful terminal output
- Async/await architecture for responsive operation
- Comprehensive test suite with pytest