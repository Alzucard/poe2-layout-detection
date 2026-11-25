"""Starts the capture from an IDE: in Visual Studio, right-click this file and pick "Set as Startup Item", then
press F5. It's the same as `python -m poe2layout`; give it the argument `analyze` to only analyze the videos left
in the video folder.

The files inside poe2layout/ can't be run on their own, because they import each other as a package."""
import runpy

runpy.run_module("poe2layout", run_name="__main__", alter_sys=True)
