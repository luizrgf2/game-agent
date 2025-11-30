#!/bin/bash

# Script to run game-agent with sudo for global hotkey support

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Este programa precisa de permissões de root para capturar hotkeys globalmente."
    echo "Executando com sudo..."
    exec sudo "$0" "$@"
fi

# Preserve user environment and run with uv
cd "$(dirname "$0")"

# Find uv in common locations
UV_PATH=""
if [ -f "$HOME/.local/bin/uv" ]; then
    UV_PATH="$HOME/.local/bin/uv"
elif [ -f "/home/$SUDO_USER/.local/bin/uv" ]; then
    UV_PATH="/home/$SUDO_USER/.local/bin/uv"
elif command -v uv &> /dev/null; then
    UV_PATH="uv"
else
    echo "Erro: uv não encontrado"
    exit 1
fi

# Run as root but with user's .env file
"$UV_PATH" run game-agent
