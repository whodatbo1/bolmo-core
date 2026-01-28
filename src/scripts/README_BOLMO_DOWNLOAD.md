# Downloading Bolmo Checkpoints

This guide explains how to download Bolmo (byte-level language model) checkpoints from HuggingFace.

## Background

### What is Bolmo?

Bolmo is a family of fully open byte-level language models at 1B and 7B parameter scales, developed by Allen AI. Unlike traditional language models that use subword tokenization, Bolmo operates directly on raw bytes (256-token vocabulary), using a hierarchical architecture with local encoder/decoder components.

### Architecture Differences: Bolmo vs OLMo

**OLMo (Standard Transformer):**
```
Tokens (subword) → Embedding → Transformer Blocks → LM Head → Logits
```

**Bolmo (Hierarchical Byte-Level):**
```
Raw Bytes (256 vocab)
    ↓
LocalEncoder (transformer blocks on bytes)
    ↓
Boundary Predictor → identifies patch boundaries
    ↓
Pooling (cross-attention or H-Net style)
    ↓
Global Transformer Blocks (operates on patches)
    ↓
LocalDecoder
    ↓
Depooling → Byte-level Representations
    ↓
LM Head → Byte Logits
```

**Key Components:**
- **LocalEncoder**: Converts bytes into patch embeddings
- **Boundary Predictor**: Identifies where patches begin/end
- **Global Transformer**: Standard transformer blocks operating on patches (not bytes)
- **LocalDecoder**: Converts patch embeddings back to byte-level predictions

### Why Can't We Reuse OLMo Download Scripts?

The existing `download_olmo_checkpoint.sh` and `convert_checkpoint_from_hf.py` scripts are designed for standard transformer architectures. They:

1. Expect a single embedding layer mapping tokens to hidden states
2. Use architecture-specific weight converters (`get_converter_from_hf()`)
3. Don't handle Bolmo's hierarchical components (LocalEncoder, LocalDecoder, boundary predictors)

**However**, Bolmo checkpoints on HuggingFace already include both:
- HuggingFace format (root directory) - for use with `transformers` library
- **OLMo-core format** (`olmo_core/` subdirectory) - native format, no conversion needed!

Our download scripts simply extract the `olmo_core/` subdirectory, avoiding any conversion.

## Available Checkpoints

| Model | HuggingFace Hub | Parameters | Architecture |
|-------|-----------------|------------|--------------|
| Bolmo-1B | [allenai/Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B) | 1 billion | Byte-level, hierarchical |
| Bolmo-7B | [allenai/Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B) | 7 billion | Byte-level, hierarchical |

## Installation Requirements

```bash
pip install huggingface_hub[cli]
```

## Usage

### Option 1: Bash Script (Simple)

```bash
# Download Bolmo-1B
bash src/scripts/download_bolmo_checkpoint.sh 1b /path/to/output

# Download Bolmo-7B
bash src/scripts/download_bolmo_checkpoint.sh 7b /path/to/output
```

### Option 2: Python Script (Recommended)

```bash
# Download Bolmo-1B
python src/scripts/download_bolmo_checkpoint.py \
    --model 1b \
    --output-dir /path/to/output

# Download Bolmo-7B with verification
python src/scripts/download_bolmo_checkpoint.py \
    --model 7b \
    --output-dir /path/to/output \
    --verify

# Download specific revision
python src/scripts/download_bolmo_checkpoint.py \
    --model 1b \
    --output-dir /path/to/output \
    --revision main
```

### Python Script Options

- `--model, -m`: Model size (`1b` or `7b`) [required]
- `--output-dir, -o`: Output directory path [required]
- `--revision, -r`: HuggingFace revision/branch (default: `main`)
- `--verify`: Verify checkpoint structure after download

## Checkpoint Structure

After downloading, the checkpoint will have the following structure:

```
output_dir/
├── config.json              # OLMo-core experiment config
├── model_and_optim/         # Model weights and optimizer state
│   ├── __0_0.distcp
│   ├── __1_0.distcp
│   ├── .metadata
│   └── ...
└── .metadata.json
```

## Using the Checkpoint

### In Training Scripts

Set the checkpoint path as an environment variable or in your training config:

```bash
export BOLMO_CKPT_PATH=/path/to/output/model_and_optim
```

### Loading in Code

```python
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from pathlib import Path

# Load config
config_path = Path("/path/to/output/config.json")
experiment_config = json.load(config_path.open())
model_config = TransformerConfig.from_dict(experiment_config["model"])

# Build model
model = model_config.build(init_device="meta")

# Load checkpoint
checkpoint_path = Path("/path/to/output/model_and_optim")
load_model_and_optim_state(checkpoint_path, model)
```

## Comparison: OLMo vs Bolmo Download Scripts

| Aspect | OLMo Script | Bolmo Script |
|--------|-------------|--------------|
| **Architecture** | Standard Transformer | Hierarchical (LocalEncoder + Global + LocalDecoder) |
| **Tokenization** | Subword (~100k vocab) | Byte-level (256 vocab) |
| **Conversion** | Uses `convert_checkpoint_from_hf.py` | No conversion needed |
| **HF Structure** | Standard HF checkpoint | HF + `olmo_core/` subdirectory |
| **Download Method** | Download + Convert | Direct extraction of `olmo_core/` |

## Troubleshooting

### Error: `huggingface-cli` not found

Install the HuggingFace Hub CLI:
```bash
pip install huggingface_hub[cli]
```

### Error: Authentication required

For private repositories, login first:
```bash
huggingface-cli login
```

### Error: Missing `olmo_core` directory

The checkpoint structure may have changed. Verify the repository structure on HuggingFace Hub:
- Check https://huggingface.co/allenai/Bolmo-1B/tree/main
- Ensure `olmo_core/` subdirectory exists

### Checkpoint verification failed

Run with `--verify` flag to see which files are missing:
```bash
python src/scripts/download_bolmo_checkpoint.py \
    --model 1b \
    --output-dir /path/to/output \
    --verify
```

## References

- [Bolmo-1B on HuggingFace](https://huggingface.co/allenai/Bolmo-1B)
- [Bolmo-7B on HuggingFace](https://huggingface.co/allenai/Bolmo-7B)
- [Bolmo Core GitHub](https://github.com/allenai/bolmo-core)
- [Bolmo Training Data](https://huggingface.co/datasets/allenai/bolmo_mix)

## License

Bolmo models are licensed under Apache 2.0 and are intended for research and educational use in accordance with Ai2's Responsible Use Guidelines.
