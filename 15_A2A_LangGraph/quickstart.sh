#!/bin/bash
# Quick start script for LangGraph Agent with A2A Protocol

echo "🚀 LangGraph Agent Quick Start"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.12 or higher."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚙️  No .env file found. Running setup..."
    python3 setup.py
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Create data directory if it doesn't exist
mkdir -p data

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Add PDF documents to the 'data' directory (optional)"
echo "2. Start the agent server:"
echo "   uv run python -m app"
echo ""
echo "3. In another terminal, test the agent:"
echo "   uv run python app/test_client.py"
echo ""
echo "📖 See README.md for more details and usage examples."
