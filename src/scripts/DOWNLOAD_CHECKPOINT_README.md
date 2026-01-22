# Downloading OLMo Checkpoints for Bolmo Training

To train Bolmo models (Stage 1 or Stage 2), you need the base OLMo checkpoint. This guide shows you how to download and convert OLMo checkpoints from HuggingFace to the olmo-core format.

## Quick Start

Use the provided download script:

```bash
# For 1B model
bash src/scripts/download_olmo_checkpoint.sh 1b /path/to/checkpoint/output

# For 7B model
bash src/scripts/download_olmo_checkpoint.sh 7b /path/to/checkpoint/output
```

## What the Script Does

1. Downloads the appropriate OLMo checkpoint from HuggingFace:
   - **1B**: `allenai/OLMo-2-0425-1B`
   - **7B**: `allenai/OLMo-2-1124-7B`

2. Converts it to olmo-core checkpoint format using `convert_checkpoint_from_hf.py`

3. Saves the converted checkpoint to your specified output directory

## Using the Checkpoint in Training

After downloading, update your training launch script to point to the checkpoint:

```bash
# In bolmo_scripts/launch_stage1_1b.sh
export OLMO_CKPT_PATH="/path/to/checkpoint/output/model_and_optim"
```

The checkpoint will be loaded when you run the training script, initializing the teacher model with pretrained OLMo weights while randomly initializing the local encoder/decoder.

## Manual Download (Alternative)

If you prefer to manually download and convert:

```bash
# Download OLMo 2 1B
python src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path allenai/OLMo-2-0425-1B \
    --output-dir /path/to/output \
    --model-arch olmo2_1b_v2 \
    --tokenizer dolma2 \
    --skip-validation

# Download OLMo 2 7B
python src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path allenai/OLMo-2-1124-7B \
    --output-dir /path/to/output \
    --model-arch olmo2_7b \
    --tokenizer dolma2 \
    --skip-validation
```

## Available Models

### OLMo 2 1B
- **HuggingFace**: [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
- **Architecture**: `olmo2_1b_v2`
- **Use with**: `OLMO_ARCH=olmo2_1B_v2` in launch scripts

### OLMo 2 7B
- **HuggingFace**: [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B)
- **Architecture**: `olmo2_7b`
- **Use with**: `OLMO_ARCH=olmo2_7B` in launch scripts

### OLMo 3 7B
- **HuggingFace**: Check [OLMo collections](https://huggingface.co/collections/allenai/olmo-2) for OLMo 3 models
- **Architecture**: `olmo3_7b`
- **Use with**: `OLMO_ARCH=olmo3_7B` in launch scripts

## Disk Space Requirements

The downloaded checkpoints require significant disk space:
- **1B model**: ~4-5 GB
- **7B model**: ~25-30 GB

Make sure you have sufficient disk space before downloading.

## Troubleshooting

### "Failed to download from HuggingFace"
- Check your internet connection
- Ensure you have the `transformers` and `huggingface_hub` packages installed
- Some models may require HuggingFace authentication (use `huggingface-cli login`)

### "Validation failed"
- Use `--skip-validation` flag to skip validation if you encounter issues
- Validation is optional and mainly useful for debugging

### "Out of memory during conversion"
- The conversion process loads the model into memory
- For large models (7B+), ensure you have at least 32GB RAM
- Use a machine with a GPU if available
