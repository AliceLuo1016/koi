#!/bin/bash
# Installation script for Koi AI Agent

echo "🐠 Installing Koi AI Agent..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3.9 or later."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --user click httpx rich tiktoken beautifulsoup4 pytest pytest-asyncio

# Create symlink to make koi command available
KOGI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_PATH="$KOGI_DIR/koi_runner.py"

# Add to PATH or suggest manual addition
if [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
    # ~/.local/bin is in PATH, create symlink there
    ln -sf "$SCRIPT_PATH" "$HOME/.local/bin/koi"
    echo "✅ Koi installed! You can now run 'koi --help'"
else
    echo "✅ Koi is ready!"
    echo "💡 To use 'koi' command globally, add this to your shell profile:"
    echo "    export PATH=\"$HOME/.local/bin:\$PATH\""
    echo "    ln -sf \"$SCRIPT_PATH\" \"$HOME/.local/bin/koi\""
    echo ""
    echo "Or run directly: python3 $SCRIPT_PATH"
fi

echo ""
echo "🚀 Quick start:"
echo "  1. cd your-project"
echo "  2. koi init"
echo "  3. Edit .koi/config.json with your API key"
echo "  4. koi run"