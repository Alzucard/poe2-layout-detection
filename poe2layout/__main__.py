"""python -m poe2layout            film zones while you play (game in Windowed Fullscreen)
python -m poe2layout analyze    analyze the videos left in the video folder, then exit"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m poe2layout", description="PoE2 layout capture")
    parser.add_argument("--version", action="version", version=f"poe2layout {__version__}")
    parser.add_argument("command", nargs="?", choices=("capture", "analyze"), default="capture")
    sys.exit(main(parser.parse_args().command))
