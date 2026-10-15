#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Render package/tt-model.yaml's card with tt-model-manager's own generator (render_model_card).

Run it with the Python of a tt-model-manager checkout that has its dependencies installed:
    ~/code/tt-model-manager/.venv/bin/python package/render_preview.py [--model-manager ~/code/tt-model-manager]
The `built` provenance the generator wants comes from the catalog package's manifest (the image this repository runs).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model-manager", default=str(Path.home() / "code" / "tt-model-manager"))
ap.add_argument("--catalog-manifest", default=str(Path.home() / ".cache/tt-model/pulled/jashansinghTT__minimax-music3-blackhole/tt_kernel_manifest.json"))
a = ap.parse_args()
sys.path.insert(0, str(Path(a.model_manager) / "src"))
from tt_kernel.build import render_model_card                       # noqa: E402
from tt_kernel.container_manifest import load_container_manifest    # noqa: E402

m = load_container_manifest(ROOT / "package" / "tt-model.yaml")
built = json.loads(Path(a.catalog_manifest).read_text())["container"]["built"]
(ROOT / "package" / "README.generated-preview.md").write_text(render_model_card(m, built))
print("wrote package/README.generated-preview.md")
