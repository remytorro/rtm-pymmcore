"""Run a notebook source headless, cell by cell, with napari stubbed out.

Exercises the real faro / inscoper_useq code paths (load, ROI, FRAP preflight,
mask pricing, validate_events, run_experiment, post-processing) without needing
a Qt event loop or a human to pick FOVs. Substitutions shrink the time plans so
a run finishes in seconds.

    python nbtest.py <source.py> [SUBST=VALUE ...]
"""

import io
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# The notebooks print box-drawing characters (utils.print_configs) and
# em-dashes, which the console default cp1252 cannot encode.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# A QApplication has to exist before any QWidget: the notebooks dock an
# ExperimentStatusWidget, which napari would have created one for.
try:
    from qtpy.QtWidgets import QApplication

    _APP = QApplication.instance() or QApplication([])
except Exception as _exc:  # noqa: BLE001
    print(f"[nbtest] no Qt ({_exc}) - widget cells will fail")
    _APP = None


class _FakeViewer:
    """Enough napari.Viewer for a notebook that only docks widgets."""

    class _Window:
        def add_dock_widget(self, *a, **k):
            return None

        dock_widgets = {}

    def __init__(self):
        self.window = self._Window()
        self.layers = {}

    def add_image(self, *a, **k):
        return None


def _stub_napari(site_module):
    site_module.open_napari = lambda mic, **k: (_FakeViewer(), None)
    site_module.relink_napari = lambda *a, **k: None
    site_module.unlink_napari = lambda *a, **k: None


def split_cells(src: str):
    cells, kind, buf = [], None, []
    for line in src.splitlines():
        if line.startswith("# %% [markdown]"):
            if kind == "code" and buf:
                cells.append("\n".join(buf))
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            if kind == "code" and buf:
                cells.append("\n".join(buf))
            kind, buf = "code", []
        else:
            buf.append(line)
    if kind == "code" and buf:
        cells.append("\n".join(buf))
    return [c for c in cells if c.strip()]


def main() -> int:
    src_path = Path(sys.argv[1])
    subs = {}
    for arg in sys.argv[2:]:
        k, _, v = arg.partition("=")
        subs[k] = v

    src = src_path.read_text(encoding="utf-8")
    for name, value in subs.items():
        # Replace a top-level assignment `NAME = ...` on one line.
        src, n = re.subn(
            rf"^{re.escape(name)} = .*$", f"{name} = {value}", src, flags=re.M
        )
        print(f"[nbtest] {name} -> {value}  ({n} site(s))")

    cells = split_cells(src)
    print(f"[nbtest] {len(cells)} code cells from {src_path.name}")

    ns: dict = {"__name__": "__main__"}
    failures = 0
    for i, cell in enumerate(cells):
        head = next((l for l in cell.splitlines() if l.strip()
                     and not l.strip().startswith("#")), "(comments only)")
        print(f"\n[nbtest] ---- cell {i}: {head[:78]}")
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = buf
        try:
            exec(compile(cell, f"{src_path.name}:cell{i}", "exec"), ns)
            status = "ok"
        except BaseException as exc:  # noqa: BLE001
            status = f"FAILED {type(exc).__name__}: {exc}"
            failures += 1
            tb = traceback.format_exc()
        else:
            tb = ""
        finally:
            sys.stdout = real_stdout
        # Keep only lines the notebook itself printed, not the C++ log spam.
        noise = re.compile(
            r"^(WARN|INFO|DEBUG|ERROR)\s|^\[\d\d:|Firmware|my_string=|^Set done"
            r"|^\t|^Device '|DCam|DCAM_|threading\.py|NumberDevice|^C14440"
            r"|^\{<inscoper|^Set new Crop|^\[STREAM\]"
        )
        for line in buf.getvalue().splitlines():
            if line.strip() and not noise.search(line):
                print("   ", line)
        print(f"[nbtest] cell {i}: {status}")
        if tb:
            print(tb)
        # Stub napari as soon as inscoper_site is in the namespace.
        if "site" in ns and hasattr(ns["site"], "open_napari"):
            _stub_napari(ns["site"])

    print(f"\n[nbtest] {len(cells) - failures}/{len(cells)} cells ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
