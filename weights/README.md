---
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
pipeline_tag: text-to-audio
base_model:
- MiniMaxAI/MiniMax-Music3
tags:
- music
- text-to-audio
- merge
- lora-merged
---

# MiniMax-Music3 with the fiona-crapple language-model adapter merged in

> Derived from [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) by MiniMax, with the language-model adapter [terminusresearch/minimax-music3-lm-lora-fiona-crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) merged into the weights. See the original model cards for training and evaluation details.

The weights of MiniMax Music 3, a lyrics-and-caption conditioned music generation model, with a rank-64 adapter merged into the language model at strength 1.0. The adapter was trained on nine tracks (about 52 minutes) in one style and answers to the trigger phrase `fiona crapple`. These weights are the input to [episod/minimax-music3-fiona-crapple-blackhole](https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole), which serves them on a Tenstorrent Blackhole chip with a Gradio demo.

## What is in this repository

The files of the base repository at revision `fbdf52fbaaca799592917417eb05f1899f1255ec` with these changes:

- The four `language_model/model-0000N-of-00004.safetensors` shards are rewritten: for each of the 252 matrices the adapter targets (the q, k, v and o projections and the gate, up and down projections of the 36 layers), `W' = W + strength * (alpha / rank) * B @ A` with strength 1.0, alpha 64 and rank 64, computed in float32 and stored as bf16. The largest change is 4.8% of a weight matrix's norm.
- Every other weight file is identical to the base's.
- `flowmatching_vae.pth` (9.8 GB), `dav.pth` and the `qwen_7B` directory of the base repository are not included. The Tenstorrent container does not load them. A stack that needs them takes them from the base repository.
- `MERGE.json` records the merge parameters.

| file | sha256 |
| --- | --- |
| `language_model/model-00001-of-00004.safetensors` | `76d78069b241644df7e117a9d433901f53c3d6cad03d007b65d5801439a519f4` |
| `language_model/model-00002-of-00004.safetensors` | `ac3e0839b7649b3faf8c852a8464b99dbb04eef96af9913a87956b8731af0302` |
| `language_model/model-00003-of-00004.safetensors` | `ab90abb94f282e8cfbbc79c1dc1e2cf32dd7ac4e3e2008a6acc599e44725cdec` |
| `language_model/model-00004-of-00004.safetensors` | `f5053b5ef2092da405ddb8c71f1a5b32939e73d188ab825ac62b917d2188430f` |

## Use

With the Tenstorrent container and demo: see [episod/minimax-music3-fiona-crapple-blackhole](https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole). These weights have been served only by that container, on one Blackhole chip. They have not been run with any other inference stack.

## Limitations

- No quality evaluation against the base model has been run. The adapter was trained on nine tracks in one style; its effect without the trigger phrase is not measured.
- Not for imitating a specific real artist's voice or presenting generated audio as a recording of a real person.

## Licensing

The MiniMax-Music3 Community License applies to these weights; products built with them must display "MiniMax-Music3". The merged adapter is Apache-2.0.
