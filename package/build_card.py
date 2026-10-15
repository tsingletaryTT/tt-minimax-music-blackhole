#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Build the Hugging Face card for this companion repository (hf/README.md).

The card text lives in the `card:` block of package/tt-model.yaml. tt-model-manager's generator renders that block into the standard
layout (package/README.generated-preview.md, made by package/render_preview.py). The generator assumes a tt-model package, which
this repository is not, so this script changes the five things that would be wrong for a companion:
  * the tags: `tt-model-catalog` and `tt-model-container` would list the repository as a package;
  * the title and the "Packaged and published" line;
  * the Quickstart, which is the block's own (the generated `tt serve <repo>` command does not work here);
  * the Provenance table, which claims `code/` in the repository is byte-identical to the image's code;
  * nothing else: every other section is the generator's output for the block.
usage: build_card.py [--preview FILE] [--yaml FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIGEST = "sha256:2da7f443622b939c3321fc092b7a0e6948911d78d72cfac53f2749564960451f"
BASE_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
CODE_DIGEST = "9a9c37114f86ca58"
TAGS = ["blackhole", "p150", "text-to-audio", "music-generation", "gradio"]


def sections(body: str) -> list[tuple[str, str]]:
    """[(heading, text)] where text excludes the heading line; the first item is the text before any `## `."""
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    out = [("", parts[0])]
    for i in range(1, len(parts), 2):
        out.append((parts[i].strip(), parts[i + 1]))
    return out


def build(preview: str, yaml_path: Path) -> str:
    card = yaml.safe_load(Path(yaml_path).read_text())["card"]
    _, fm, body = preview.split("---", 2)
    fm = re.sub(r"tags:\n(?:- .+\n)+", "tags:\n" + "".join(f"- {t}\n" for t in TAGS), fm, count=1)
    secs = sections(body)
    head = secs[0][1]
    head = re.sub(r"^# .+$", "# minimax-music3-fiona-crapple on Tenstorrent p150 (companion)", head, count=1, flags=re.M)
    head = re.sub(r"Packaged and published with .+\n", "Companion repository for the "
                  "[minimax-music3-blackhole](https://huggingface.co/jashansinghTT/minimax-music3-blackhole) container package: "
                  "a launcher that serves it with the merged weights, and a Gradio demo.\n", head, count=1)
    out = ["---" + fm + "---" + head.rstrip("\n") + "\n"]
    for name, text in secs[1:]:
        if name == "Quickstart":
            text = "\n\n" + card["quickstart"].strip("\n") + "\n\n"
        elif name == "Provenance":
            text = f"""

What this repository runs:

| component | source |
| --- | --- |
| container image | `{IMAGE_DIGEST}`, from [jashansinghTT/minimax-music3-blackhole](https://huggingface.co/jashansinghTT/minimax-music3-blackhole) (model code digest `{CODE_DIGEST}`, sha256, first 16 hex digits) |
| base weights | [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) at `{BASE_REVISION}` |
| adapter | [terminusresearch/minimax-music3-lm-lora-fiona-crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple), `regularised/checkpoint-750`, rank 64, alpha 64 |
| merge | `W' = W + strength * (alpha / rank) * B @ A` on the 252 language-model matrices the adapter targets, strength 1.0, stored in bf16; every other weight file is the base's (`serve/merge_lora.py`) |
"""
        out.append(f"## {name}{text}")
    return "".join(out).rstrip("\n") + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", default=str(ROOT / "package" / "README.generated-preview.md"))
    ap.add_argument("--yaml", default=str(ROOT / "package" / "tt-model.yaml"))
    ap.add_argument("--out", default=str(ROOT / "hf" / "README.md"))
    a = ap.parse_args(argv)
    text = build(Path(a.preview).read_text(), Path(a.yaml))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
