# Koi Agent Example

Here's a quick example of using the Koi AI agent:

## 1. Installation

```bash
cd ~/koi
./install.sh
```

## 2. Initialize a project

```bash
mkdir my-project
cd my-project
koi init
```

## 3. Configure API settings

Edit `.agent/config.json`:

```json
{
  "api_base": "https://api.openai.com/v1",
  "api_key": "your-actual-api-key-here",
  "model": "gpt-5.2",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": ["./skills"],
  "temperature": 0.7
}
```

## 4. Start the agent

```bash
koi run
```

## Example conversation:

```
🐠 Koi Agent - Ready to help!
Type '/exit' to quit, '/help' for commands

koi> What files are in this directory?
🔧 exec_command...
The current directory contains:
- .agent/ (directory containing config, memory, etc.)

koi> Create a simple Python hello world script
🔧 write_file...
I've created a file called hello.py with a simple "Hello, World!" script.

koi> /remember This project is for testing the Koi agent
✅ Added to memory

koi> /exit
👋 Goodbye!
```

## 5. Schedule tasks

```bash
# Check files every hour
koi cron add "0 * * * *" "List files in the current directory and save to daily-log.txt"

# Daily summary
koi cron add "0 18 * * *" "Create a summary of today's work and append to weekly-report.md"
```

## Available commands:

- `koi run` - Interactive session
- `koi init` - Initialize .agent directory
- `koi cron add "schedule" "task"` - Schedule tasks
- `koi cron list` - List scheduled tasks
- `koi skills` - List available skills
- `koi config` - Show configuration
- `koi memory` - Show current memory

## Chat commands (during `koi run`):

- `/memory` - Show memory
- `/remember TEXT` - Add to memory
- `/skills` - List skills
- `/stats` - Show token usage
- `/compact` - Compress conversation
- `/exit` - Quit