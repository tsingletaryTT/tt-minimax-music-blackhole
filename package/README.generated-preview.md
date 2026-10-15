---
tags:
- blackhole
- p150
- tt-dit-server
- tt-model-cache
- tt-model-container
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
pipeline_tag: text-to-audio
base_model:
- MiniMaxAI/MiniMax-Music3
---

# minimax-music3-fiona-crapple on Tenstorrent p150

> Derived from [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) by MiniMax, with the language-model adapter [terminusresearch/minimax-music3-lm-lora-fiona-crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) (`regularised/checkpoint-750`) merged into the weights at strength 1.0. See the original model cards for training and evaluation details.

MiniMax Music 3, a lyrics-and-caption conditioned music generation model (8B Qwen3-based language model, 0.6B depth decoder for the residual RVQ codebooks, 2.4B flow-matching DiT and a vocoder), with a rank-64 adapter merged into the language model. The adapter was trained on nine tracks (about 52 minutes) in one style and answers to the trigger phrase `fiona crapple`. The server makes one song at a time and returns 32 kHz stereo audio through the model's own HTTP job API.

Runs on Tenstorrent **p150** (mesh `P150`).

Packaged and published with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 5.1).

## At a glance

| | |
| --- | --- |
| Hardware | p150 |
| License | minimax-music3-community-license |

## Intended use

Generating songs from lyrics and a text description of the music. Putting `fiona crapple` in the description steers the output toward the adapter's training style, which its author describes as an intimate cafe singer-songwriter sound with jazzy piano, upright bass, brushed drums and a raw female vocal.

## Prerequisites

- The [Tenstorrent CLI](https://github.com/tenstorrent/tt-cli/blob/main/docs/prerequisites.md), `tt`: `uv tool install tenstorrent` (or use tt-model alone, see the Quickstart)
- Docker
- Tenstorrent hardware: **p150**
- About 27 GB of free disk for the merged weights and 4 GB for the container image
- Python 3.12 for the demo (`pip install -r requirements.txt`)

## Quickstart

```bash
tt serve episod/minimax-music3-fiona-crapple-blackhole
```

`tt serve` downloads the Docker image and the [`episod/minimax-music3-fiona-crapple`](https://huggingface.co/episod/minimax-music3-fiona-crapple) weights (into your HF cache; they are not in the image), then starts the model's own HTTP server on port 20000 (or the next free port, if that one is busy). The first start compiles kernels for your device, which takes several minutes; the server is ready when it logs `Application startup complete`.

Without tt-cli: `tt-model serve episod/minimax-music3-fiona-crapple-blackhole`.

This repository is a companion, not a tt-model package: it holds the launcher and a Gradio demo, and the weights are in
[episod/minimax-music3-fiona-crapple](https://huggingface.co/episod/minimax-music3-fiona-crapple).

```bash
git clone https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole
cd minimax-music3-fiona-crapple-blackhole
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
gozer run --chips 1 -- .venv/bin/python serve/serve_music.py --ui     # without gozer: drop `gozer run --chips 1 --`
```

The launcher installs the catalog image (`tt-model pull jashansinghTT/minimax-music3-blackhole`) if it is missing, downloads the
weights, starts the server on port 20000 and the Gradio page on port 7871.

## Capabilities

This package exposes the model's own HTTP server on port 20000, **not** an OpenAI-compatible chat API — its request and response shapes are the model's own. See the author's notes for the payload it expects.

```bash
curl -s localhost:20000/v1/music/jobs -H 'Content-Type: application/json' -d '{
  "input": "[Verse 1]\nThe kettle sings at half past three\n[Chorus]\nStay a little longer",
  "instructions": "fiona crapple, jazzy piano, raw female vocal, upright bass, brushed drums, intimate",
  "audio_duration": 20, "seed": 11, "response_format": "flac"}'
# -> {"id": "...", "status": "queued", "queue_position": 0}
curl -s localhost:20000/v1/music/jobs/<id>            # status, frames_done, stats when done
curl -s localhost:20000/v1/music/jobs/<id>/audio -o song.flac
```

`audio_duration` is in seconds (25 audio frames per second). The same seed, lyrics and description gave the same file: a repeated
request on one server, and the same request on a server on the other board, each returned a bit-identical file (one pair each).
In a test, 9 jobs submitted at once were accepted and a tenth got HTTP 429.

## Expected performance

| eval | this package | reference | prompts |
| --- | --- | --- | --- |
| Quality evaluation against MiniMax-Music3 | not run | not run | 0 |

| profile | audio length | concurrency | N | compute time (s) | per second of audio (s) | language model (s) | depth decoder (s) | diffusion (s) | vocoder (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `p150` | 20 s | 1 | 20 | 65.6 | 3.3 | 18.2 | 19.2 | 24.1 | 16.0 |

Medians of the server's own job timings for 20 requests, one song at a time, on one chip of a p300c board (QuietBox 2). The compute
time ranged from 64.8 to 68.7 s. One request finished at 16.6 s because the server reported the song as ended. The server was
healthy 90 to 172 s after its container started (4 starts).

## Limitations

- One song is generated at a time. In a test, 9 jobs were accepted at once (one running, eight waiting) and a tenth got HTTP 429.
- Slower than real time: about 3.3 s of compute per second of audio on one chip.
- Only the `p150` profile on one chip is tested, and only with 20 s requests. The server allows up to 9000 frames (6 minutes).
- A song can end before the requested length when the lyrics are finished (1 of 20 test requests: 16.6 s of 20 s).
- No quality evaluation against the base model has been run. The adapter was trained on nine tracks in one style; its effect without
  the trigger phrase is not measured.
- Not for imitating a specific real artist's voice or presenting generated audio as a recording of a real person.
- On a host where loading a cached tensor of about 32 MiB or more never returns (seen with Linux 7.0.0-38, KMD 2.11.0), the launcher
  runs the container with its tensor cache disabled.
- The server has no authentication; run it only on a network you trust.

## Licensing

Model weights: MiniMax-Music3 Community License; products built with the weights must display "MiniMax-Music3". The merged adapter is
Apache-2.0. Launcher and demo code: Apache-2.0.

## Related packages

- [jashansinghTT/minimax-music3-blackhole](https://huggingface.co/jashansinghTT/minimax-music3-blackhole): the unmodified MiniMax-Music3 container package this repository's launcher runs.
- [episod/minimax-music3-fiona-crapple](https://huggingface.co/episod/minimax-music3-fiona-crapple): the merged weights.

## Feedback

Questions or problems with this package: open a discussion at https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole/discussions — that is what reaches its author. A problem with the `tt` tooling itself: `tt report issue` (collects your environment and opens a prefilled issue against tenstorrent/tt-cli). Product feedback: support@tenstorrent.com.

## Provenance

The exact sources the image was built from — `code/` in this repo is byte-identical to the model code inside the image:

| component | built from |
| --- | --- |
| tt-metal | a local checkout — commit not published |
| `code/` digest | `9a9c37114f86ca58` (sha256, first 16 hex digits) |
| built | 2026-09-12T09:31:21+00:00 by tt-model 0.1.0 |

