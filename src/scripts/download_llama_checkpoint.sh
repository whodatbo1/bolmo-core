#!/bin/bash
# Download and convert Llama checkpoint from HuggingFace for Bolmo training
#
# Usage:
#   bash src/scripts/download_llama_checkpoint.sh llama2-7b /path/to/output
#   bash src/scripts/download_llama_checkpoint.sh llama3.1-70b /path/to/output
#
# Supported models:
#   Llama 2: llama2-7b, llama2-13b, llama2-70b
#   Llama 3: llama3-8b, llama3-70b
#   Llama 3.1: llama3.1-8b, llama3.1-70b, llama3.1-405b
#   Llama 3.2: llama3.2-1b, llama3.2-3b

set -e

MODEL_SIZE=$1
OUTPUT_DIR=$2
TOKENIZER=${3:-"gpt2"}  # Default to gpt2, can be overridden

if [ -z "$MODEL_SIZE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <model_size> <output_dir> [tokenizer]"
    echo ""
    echo "Supported models:"
    echo "  Llama 2:   llama2-7b, llama2-13b, llama2-70b"
    echo "  Llama 3:   llama3-8b, llama3-70b"
    echo "  Llama 3.1: llama3.1-8b, llama3.1-70b, llama3.1-405b"
    echo "  Llama 3.2: llama3.2-1b, llama3.2-3b"
    echo ""
    echo "Arguments:"
    echo "  model_size:  One of the supported model identifiers above"
    echo "  output_dir:  Directory where checkpoint will be saved"
    echo "  tokenizer:   (Optional) Tokenizer to use (default: gpt2)"
    echo "               Options: dolma2, gpt2, gpt_neox_olmo_dolma_v1_5"
    exit 1
fi

case $MODEL_SIZE in
    # Llama 2 models
    llama2-7b)
        HF_MODEL="meta-llama/Llama-2-7b-hf"
        MODEL_ARCH="llama2_7b"
        echo "Downloading Llama 2 7B checkpoint..."
        ;;
    llama2-13b)
        HF_MODEL="meta-llama/Llama-2-13b-hf"
        MODEL_ARCH="llama2_13b"
        echo "Downloading Llama 2 13B checkpoint..."
        ;;
    llama2-70b)
        HF_MODEL="meta-llama/Llama-2-70b-hf"
        MODEL_ARCH="llama2_70b"
        echo "Downloading Llama 2 70B checkpoint..."
        ;;

    # Llama 3 models
    llama3-8b)
        HF_MODEL="meta-llama/Meta-Llama-3-8B"
        MODEL_ARCH="llama3_8b"
        echo "Downloading Llama 3 8B checkpoint..."
        ;;
    llama3-70b)
        HF_MODEL="meta-llama/Meta-Llama-3-70B"
        MODEL_ARCH="llama3_70b"
        echo "Downloading Llama 3 70B checkpoint..."
        ;;

    # Llama 3.1 models
    llama3.1-8b)
        HF_MODEL="meta-llama/Meta-Llama-3.1-8B"
        MODEL_ARCH="llama3_8b"
        echo "Downloading Llama 3.1 8B checkpoint..."
        ;;
    llama3.1-70b)
        HF_MODEL="meta-llama/Meta-Llama-3.1-70B"
        MODEL_ARCH="llama3_70b"
        echo "Downloading Llama 3.1 70B checkpoint..."
        ;;
    llama3.1-405b)
        HF_MODEL="meta-llama/Meta-Llama-3.1-405B"
        MODEL_ARCH="llama3_405b"
        echo "Downloading Llama 3.1 405B checkpoint..."
        ;;

    # Llama 3.2 models
    llama3.2-1b)
        HF_MODEL="meta-llama/Llama-3.2-1B"
        MODEL_ARCH="llama3_1b"
        echo "Downloading Llama 3.2 1B checkpoint..."
        ;;
    llama3.2-3b)
        HF_MODEL="meta-llama/Llama-3.2-3B"
        MODEL_ARCH="llama3_8b"  # Using 8B arch as placeholder, may need adjustment
        echo "Downloading Llama 3.2 3B checkpoint..."
        echo "WARNING: Using llama3_8b architecture as placeholder for 3B model."
        echo "         You may need to verify this is correct for your use case."
        ;;

    *)
        echo "Error: Unsupported model size '$MODEL_SIZE'"
        echo ""
        echo "Supported models:"
        echo "  Llama 2:   llama2-7b, llama2-13b, llama2-70b"
        echo "  Llama 3:   llama3-8b, llama3-70b"
        echo "  Llama 3.1: llama3.1-8b, llama3.1-70b, llama3.1-405b"
        echo "  Llama 3.2: llama3.2-1b, llama3.2-3b"
        exit 1
        ;;
esac

echo "HuggingFace Model: $HF_MODEL"
echo "Bolmo Architecture: $MODEL_ARCH"
echo "Output Directory: $OUTPUT_DIR"
echo "Tokenizer: $TOKENIZER"
echo ""
echo "NOTE: You will need HuggingFace access to Meta Llama models."
echo "      Request access at: https://huggingface.co/meta-llama"
echo "      Then authenticate using one of:"
echo "        1. huggingface-cli login"
echo "        2. export HF_TOKEN=your_token_here"
echo ""

# Run the conversion script
python $BOLMO_HOME/src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path "$HF_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --model-arch "$MODEL_ARCH" \
    --tokenizer "$TOKENIZER" \
    --skip-validation

echo ""
echo "✓ Checkpoint downloaded and converted successfully!"
echo "Set OLMO_CKPT_PATH=$OUTPUT_DIR/model_and_optim in your launch script"
