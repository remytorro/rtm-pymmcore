"""Turn a `# %%` / `# %% [markdown]` annotated .py into a .ipynb.

Inverse of the nb2py extractor used to read the originals, so a notebook can
be authored and reviewed as plain Python and shipped as JSON.
"""

import json
import sys
from pathlib import Path


def convert(src_path: Path, out_path: Path) -> int:
    lines = src_path.read_text(encoding="utf-8").splitlines()
    cells = []
    kind = None
    buf: list[str] = []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            body = "\n".join(
                l[2:] if l.startswith("# ") else (l[1:] if l == "#" else l)
                for l in text.splitlines()
            )
            cells.append(
                {"cell_type": "markdown", "metadata": {},
                 "source": [f"{l}\n" for l in body.splitlines()]}
            )
        else:
            cells.append(
                {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "source": [f"{l}\n" for l in text.splitlines()]}
            )

    for line in lines:
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "py313",
                "language": "python",
                "name": "py313",
            },
            "language_info": {"name": "python", "version": "3.13.7"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return len(cells)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    n = convert(src, out)
    print(f"{out}  ({n} cells)")
