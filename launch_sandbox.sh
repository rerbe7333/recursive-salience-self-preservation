#!/bin/bash

# Interactive Sandbox Launcher
# Simple script to start the recursive salience sandbox

echo "🤖 Recursive Salience Interactive Sandbox"
echo "=========================================="
echo ""

# Check if requirements are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "⚠️  Installing required dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "🚀 Starting interactive sandbox..."
echo "📍 Server will be available at: http://localhost:7860"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Launch the sandbox
python interactive_sandbox.py
