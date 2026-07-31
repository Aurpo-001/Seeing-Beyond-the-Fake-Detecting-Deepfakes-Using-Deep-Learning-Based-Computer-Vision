#!/usr/bin/env python3
"""Validate the public notebook without importing its ML dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "deepfake_detection_pipeline.ipynb"
FORBIDDEN_PATHS = ("/mnt/", "/Users/", "C:\\")
REQUIRED_MARKERS = (
    "Video-first 70/20/10 split",
    "Proposed stronger custom CNN backbone",
    "Extract video clips",
    "Precompute custom CNN feature sequences",
    "build_bilstm_temporal_model",
    "Final video-level fusion",
    "Grad-CAM",
)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if cell.get("outputs"):
            errors.append(f"cell {index}: stored outputs must be empty")
        try:
            ast.parse("".join(cell.get("source", [])))
        except SyntaxError as exc:
            errors.append(f"cell {index}: {exc}")

    for path in FORBIDDEN_PATHS:
        if path in source:
            errors.append(f"machine-specific path remains: {path}")
    for marker in REQUIRED_MARKERS:
        if marker not in source:
            errors.append(f"required stage missing: {marker}")

    if errors:
        raise SystemExit("Notebook validation failed:\n- " + "\n- ".join(errors))
    print(f"Validated {NOTEBOOK.relative_to(ROOT)} ({len(notebook['cells'])} cells, no stored outputs).")


if __name__ == "__main__":
    main()
