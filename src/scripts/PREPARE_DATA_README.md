# Preparing Bolmo Training Data

This guide explains how to download and prepare the `bolmo_mix` dataset for training.

## Quick Start

The `prepare_bolmo_data.py` script downloads a portion of the bolmo_mix dataset from HuggingFace, tokenizes it, and saves it as `.npy` files ready for training.

### Basic Usage

Download 1% of the data (good for testing):

```bash
python src/scripts/prepare_bolmo_data.py \
    --output-dir /path/to/data/output \
    --data-fraction 0.01 \
    --tokenizer byte
```

Download 10% of the data for more substantial training:

```bash
python src/scripts/prepare_bolmo_data.py \
    --output-dir /path/to/data/output \
    --data-fraction 0.1 \
    --tokenizer byte
```

### Options

- `--output-dir`: Directory where .npy files and data_sources.txt will be saved (required)
- `--data-fraction`: Fraction of data to download, between 0.0 and 1.0 (default: 0.01)
- `--tokenizer`: Tokenizer type - either `byte` or `subword` (default: byte)
- `--tokenizer-identifier`: HuggingFace tokenizer identifier (default: allenai/dolma2-tokenizer)
- `--max-tokens-per-file`: Maximum tokens per .npy file (default: 100M)
- `--seed`: Random seed for sampling (default: 42)
- `--no-streaming`: Disable streaming mode (not recommended for large datasets)

### Output

The script creates:

1. **`.npy` files**: One or more numpy array files containing tokenized data
   - Format: `bolmo_data_00000.npy`, `bolmo_data_00001.npy`, etc.
   - Data type: uint32
   - Each file contains up to `--max-tokens-per-file` tokens

2. **`data_sources.txt`**: A text file listing absolute paths to all .npy files
   - One path per line
   - This file is used as the `DATA_SOURCE` in training scripts

### Using the Prepared Data

After running the script, you can use the prepared data in training:

```bash
# In your launch script or environment:
export DATA_SOURCE=/path/to/data/output/data_sources.txt

# Then run training:
bash bolmo_scripts/launch_stage1_1b.sh
```

Or modify the launch script to set:

```bash
DATA_SOURCE=/path/to/data/output/data_sources.txt
```

## Examples

### Example 1: Minimal test dataset (0.1% of data)

```bash
python src/scripts/prepare_bolmo_data.py \
    --output-dir ./data/bolmo_test \
    --data-fraction 0.001 \
    --tokenizer byte
```

This will download approximately 170M tokens (~0.1% of 172B total).

### Example 2: Small training dataset (5% of data)

```bash
python src/scripts/prepare_bolmo_data.py \
    --output-dir ./data/bolmo_5pct \
    --data-fraction 0.05 \
    --tokenizer byte \
    --max-tokens-per-file 200000000
```

This will download approximately 8.6B tokens (~5% of 172B total).

### Example 3: Using subword tokenizer instead of byte tokenizer

```bash
python src/scripts/prepare_bolmo_data.py \
    --output-dir ./data/bolmo_subword \
    --data-fraction 0.01 \
    --tokenizer subword \
    --tokenizer-identifier allenai/dolma2-tokenizer
```

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install datasets transformers tqdm
```

These should already be installed if you've followed the main installation instructions in the repository README.

## Notes

- **Streaming mode** (default): Recommended for large datasets. Downloads and processes data on-the-fly without loading everything into memory.
- **Data fraction**: The actual number of examples processed may vary slightly due to estimation in streaming mode.
- **Disk space**: Each .npy file is approximately 400MB per 100M tokens. Plan accordingly.
- **Time**: Processing time depends on your internet speed and CPU. Expect ~1-2 hours for 1% of data.

## Troubleshooting

### "datasets library not found"

Install it with:
```bash
pip install datasets
```

### "Failed to load dataset"

Make sure you have:
- Internet connection
- Sufficient disk space
- Access to HuggingFace (no authentication required for public datasets)

### Script interrupted

The script handles interruptions gracefully (Ctrl+C). It will save any progress made so far.

### Memory issues

If you encounter memory issues, make sure streaming mode is enabled (it's on by default). If you used `--no-streaming`, remove that flag.
