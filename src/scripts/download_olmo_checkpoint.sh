#!/bin/bash
# Download and convert OLMo checkpoint from HuggingFace for Bolmo training
#
# Usage:
#   bash src/scripts/download_olmo_checkpoint.sh 1b /path/to/output
#   bash src/scripts/download_olmo_checkpoint.sh 7b /path/to/output

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
        HF_MODEL="allenai/OLMo-2-0425-1B"
        MODEL_ARCH="olmo2_1b_v2"
        echo "Downloading OLMo 2 1B checkpoint..."
        ;;
    7b)
        HF_MODEL="allenai/OLMo-2-1124-7B"
        MODEL_ARCH="olmo2_7b"
        echo "Downloading OLMo 2 7B checkpoint..."
        ;;
    *)
        echo "Error: model_size must be '1b' or '7b'"
        exit 1
        ;;
esac

echo "HuggingFace Model: $HF_MODEL"
echo "OLMo-core Architecture: $MODEL_ARCH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Run the conversion script
python src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path "$HF_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --model-arch "$MODEL_ARCH" \
    --tokenizer dolma2 \
    --skip-validation

echo ""
echo "✓ Checkpoint downloaded and converted successfully!"
echo "Set OLMO_CKPT_PATH=$OUTPUT_DIR/model_and_optim in your launch script"
