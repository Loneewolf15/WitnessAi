"""Pytest configuration — sets up Python path for imports."""
import sys
import os

# Add the witnessai directory to path so all imports work
sys.path.insert(0, os.path.dirname(__file__))
