#!/usr/bin/env python
"""NutriCHAT — 家禽营养 AI Agent 命令行入口。

Usage:
    python main.py --query "What is the crude protein of Corn Grain?"
    python main.py --interactive
    python main.py --evaluate
    python main.py --compare
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
