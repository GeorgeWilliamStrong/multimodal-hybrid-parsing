import subprocess
import sys
from .page_images import PageImageConverter
from .heuristic_parser import DocumentParser


def _check_system_dependencies():
    """Check if required system dependencies are installed."""
    missing = []
    
    # Check LibreOffice
    try:
        subprocess.run(
            ['soffice', '--version'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("LibreOffice (soffice)")
    
    # Check Poppler (pdf2image dependency)
    try:
        subprocess.run(
            ['pdftoppm', '-v'],
            capture_output=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("Poppler (pdftoppm)")
    
    if missing:
        print("Warning: Missing system dependencies:", file=sys.stderr)
        print("Please install:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nSee README.md for installation instructions.", file=sys.stderr)


# Run dependency check on import
_check_system_dependencies()

__all__ = ['PageImageConverter', 'DocumentParser']
