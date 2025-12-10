<div align="center">
  <br>
  <h1>Bolmo</h1>
  <h4>The first fully open byte-level language model performing on par with state-of-the-art subword models</h4>
</div>

<p align="center">
  <a href="https://github.com/allenai/bolmo-core">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github"></a>
  <a href="https://huggingface.co/collections/allenai/bolmo">
    <img alt="HuggingFace Models" src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://github.com/allenai/bolmo-core/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9"></a>
</p>

---

**Bolmo** is the first fully-open byte-level language model achieving performance on par with or surpassing state-of-the-art subword-level language models. Unlike traditional language models that rely on subword tokenizers (like BPE or WordPiece), Bolmo operates directly on raw UTF-8 bytes, making it:

- **Free of subword tokenization**: No need for language-specific tokenizers or vocabulary management
- **Universally applicable**: Works seamlessly across all languages, scripts, and domains
- **Fully open**: Complete training code, model weights, data processing pipeline, and paper
- **Competitive performance**: Comes close to matching (and in some cases exceeds) subword-based state-of-the-art models across a wide range of tasks
- **Better character understanding**: Superior performance on tasks requiring character-level knowledge

This repository is a fork of [OLMo-core](https://github.com/allenai/OLMo-core) that implements the complete Bolmo architecture and training pipeline through **byteifying** - our approach to converting existing subword models to byte-level models, using <1% of the pretraining budget.

## Models

We release Bolmo models in two sizes:

| Model | Parameters | Base Model | HuggingFace |
|-------|-----------|------------|-------------|
| **Bolmo-7B** | 7.6B | Olmo 3 7B | [allenai/Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B) |
| **Bolmo-1B** | 1.5B | OLMo 2 1B | [allenai/Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B) |

**Dataset**: Training data based on Dolma 3 pretraining mix + StackEdu code data + CUTE-style character understanding tasks.

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

### From Source (Recommended for Development)

```bash
git clone https://github.com/allenai/bolmo-core.git
cd bolmo-core
pip install -e .[all]
```

### Optional Dependencies

For full functionality, you may need:
- [flash-attn](https://github.com/Dao-AILab/flash-attention) for efficient attention
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for optimized training
- [xlstm](https://github.com/NX-AI/xlstm) for xLSTM components (mLSTM layers used in the Bolmo local models)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for low-memory loss implementations

See the [OLMo-core documentation](https://olmo-core.readthedocs.io/) for complete installation details.

## Quick Start

### Inference with HuggingFace

```python
TODO
```

Or with the pipeline API:

```python
TODO
```

## Training

Bolmo training uses a two-stage "byteifying" procedure to convert existing subword models to byte-level:

### Stage 1: Subword-to-Byte Distillation
Quickly learn weights for local models while freezing the global model (9.8B tokens ≈ 43B bytes).

### Stage 2: End-to-End Training
Train the entire model to utilize byte-level information (39.3B tokens ≈ 173B bytes).

### Example Training Command

```bash
# Stage 1
TODO

# Stage 2 (after Stage 1 completes)
TODO
```

See [`src/examples/bolmo/`](src/examples/bolmo/) for detailed training scripts and configuration options.

## Architecture

Bolmo uses a novel architecture that enables converting subword models to efficient byte-level language models:

TODO: image

## Key Features

### 1. Universal Language Support
No vocabulary limitations - works seamlessly across all languages, scripts, and domains without language-specific tokenizers.

### 2. Superior Character Understanding
Achieves 78.6% on CUTE (vs 56.9% for Olmo 3) and 71.6% on EXECUTE benchmarks through dedicated character-level training data.

### 3. Adjustable Compression
Unlike subword models, Bolmo can arbitrarily adjust the bytes-per-patch ratio to trade off speed for performance:

```python
# Train with higher compression for faster inference
torchrun --nproc-per-node=8 src/examples/bolmo/train_stage2.py \
  --target-compression=8.0  # vs default ~4.4
```

### 4. Zero-Cost Post-Training
Existing post-trained checkpoints can be byteified without additional training using Task Arithmetic:

```python
from olmo_core.nn.bolmo import byteify_checkpoint

# Merge post-trained checkpoint into Bolmo
byteified_model = byteify_checkpoint(
    bolmo_base="allenai/Bolmo-7B",
    posttrain_checkpoint="allenai/OLMo-3-7B-Instruct"
)
```

### 5. Efficient Training
Total training cost: only 39.3B tokens (≈173B bytes) to byteify an existing model - orders of magnitude less than training from scratch.

## Performance

### Bolmo 7B Results

Bolmo 7B comes to matches or exceeds the performance of state-of-the-art byte-level models and comes close to the source Olmo 3 7B model:

| Category | Bolmo 7B | Olmo 3 7B | BLT 7B |
|----------|----------|-----------|---------|
| Character Understanding (CUTE) | 78.6 | 56.9 | 52.3 |
| Multilingual Char (EXECUTE) | 71.6 | 55.1 | 46.3 |
| Code | 41.0 | 40.1 | - |
| Math | 48.9 | 55.3 | - |
| MC Stem | 65.5 | 66.3 | 49.0 |
| MC Non-Stem | 75.8 | 77.7 | 56.6 |
| GenQA | 70.9 | 72.4 | 68.4 |

Full evaluation results available in the paper.

## Citation

If you use Bolmo in your research, please cite:

```bibtex
<Citation info forthcoming!>
```

For the underlying OLMo-core framework:

```bibtex
@misc{olmo20242olmo2furious,
  title={{2 OLMo 2 Furious}},
  author={{Team OLMo} and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
  year={2024},
  eprint={2501.00656},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.00656},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
