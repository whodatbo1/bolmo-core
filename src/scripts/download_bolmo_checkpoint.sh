#!/bin/bash
# Download Bolmo checkpoint from HuggingFace for training/fine-tuning
#
# The Bolmo checkpoints on HuggingFace include both:
# 1. HuggingFace format (in root directory)
# 2. OLMo-core format (in olmo_core/ subdirectory)
#
# This script downloads the olmo-core format which can be directly loaded
# by bolmo-core without any conversion needed.
#
# Usage:
#   bash src/scripts/download_bolmo_checkpoint.sh 1b /path/to/output
#   bash src/scripts/download_bolmo_checkpoint.sh 7b /path/to/output

set -e

MODEL_SIZE=$1
OUTPUT_DIR=$2

if [ -z "$MODEL_SIZE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <model_size> <output_dir>"
    echo "  model_size: 1b or 7b"
    echo "  output_dir: Directory where checkpoint will be saved"
    exit 1
fi

case $MODEL_SIZE in
    1b)
        HF_MODEL="allenai/Bolmo-1B"
        echo "Downloading Bolmo 1B checkpoint..."
        ;;
    7b)
        HF_MODEL="allenai/Bolmo-7B"
        echo "Downloading Bolmo 7B checkpoint..."
        ;;
    *)
        echo "Error: model_size must be '1b' or '7b'"
        exit 1
        ;;
esac

echo "HuggingFace Model: $HF_MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli is not installed."
    echo "Install it with: pip install huggingface_hub[cli]"
    exit 1
fi

echo "Downloading olmo-core checkpoint from HuggingFace..."
echo "Note: The Bolmo HF checkpoints include a pre-converted olmo-core format"
echo "      in the 'olmo_core/' subdirectory, so no conversion is needed."
echo ""

# Download the olmo_core subdirectory from the HF repo
# This contains the model_and_optim directory and config.json in olmo-core format
huggingface-cli download "$HF_MODEL" \
    --include "olmo_core/*" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

# Move files from olmo_core subdirectory to root of output directory
if [ -d "$OUTPUT_DIR/olmo_core" ]; then
    echo "Extracting olmo-core checkpoint..."
    mv "$OUTPUT_DIR/olmo_core"/* "$OUTPUT_DIR/"
    rmdir "$OUTPUT_DIR/olmo_core"
fi

echo ""
echo "✓ Checkpoint downloaded successfully!"
echo ""
echo "The checkpoint is now available at: $OUTPUT_DIR"
echo "- Model weights: $OUTPUT_DIR/model_and_optim/"
echo "- Config: $OUTPUT_DIR/config.json"
echo ""
echo "To use this checkpoint in training, set:"
echo "  BOLMO_CKPT_PATH=$OUTPUT_DIR/model_and_optim"
