# Changelog

What changed in each release. Versions follow [SemVer](https://semver.org): `1.0.1` fixes bugs, `1.1.0` adds
features, `2.0.0` breaks old `config.json` settings or saved runs. New changes go under "Unreleased" until the
next release (see "Releases" in the README).

## Unreleased

- `python -m poe2layout --version` shows the version, and the log names it at every start.

## 1.0.0 - 2026-11-25

First release.

- Films zones while you play and saves each zone instance as a run: the stitched Tab map, the route you walked
  and the map labels (read with OCR from the video, after the zone is finished).
- Groups the runs of a zone into layouts, and handles zones with several floors.
- Recording starts and stops with ctrl+alt+r (`recorder.manual: false` films every zone instead); ctrl+alt+p
  pauses without ending the video, ctrl+alt+q quits.
- `analyze.bat` analyzes the videos left in the video folder.
- MIT license.
