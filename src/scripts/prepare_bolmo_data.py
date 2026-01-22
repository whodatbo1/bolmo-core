#!/usr/bin/env python3
"""
Download and prepare bolmo_mix dataset for training.

This script downloads a portion of the bolmo_mix dataset from HuggingFace,
tokenizes it using either byte or subword tokenizer, and saves it as .npy files
ready for training.

Example usage:
    # Download 1% of the data using byte tokenizer
    python src/scripts/prepare_bolmo_data.py \
        --output-dir /path/to/output \
        --data-fraction 0.01 \
        --tokenizer byte \
        --max-tokens-per-file 100000000

    # Download 10% using subword tokenizer (dolma2)
    python src/scripts/prepare_bolmo_data.py \
        --output-dir /path/to/output \
        --data-fraction 0.1 \
        --tokenizer dolma2 \
        --max-tokens-per-file 100000000
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install it with: pip install datasets")
    sys.exit(1)

from transformers import AutoTokenizer

# Add the src directory to the path to import olmo_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo_core.data.tokenizer import TokenizerConfig, ByteTokenizerConfig, ByteTokenizer
from olmo_core.data.utils import write_array_to_disk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def create_byte_tokenizer(base_tokenizer: str = "allenai/dolma2-tokenizer") -> ByteTokenizer:
    """Create a byte tokenizer from a base subword tokenizer."""
    base_config = TokenizerConfig.dolma2()
    byte_config = ByteTokenizerConfig.from_tokenizer_config(base_config)
    return ByteTokenizer(byte_config)


def tokenize_text(
    text: str,
    tokenizer_type: str,
    tokenizer_identifier: str = "allenai/dolma2-tokenizer"
) -> np.ndarray:
    """
    Tokenize text using specified tokenizer.

    Args:
        text: Text to tokenize
        tokenizer_type: Either 'byte' or 'subword'
        tokenizer_identifier: HuggingFace identifier for the tokenizer

    Returns:
        Array of token IDs
    """
    if tokenizer_type == "byte":
        byte_tokenizer = create_byte_tokenizer(tokenizer_identifier)
        # For byte tokenizer, we need to first get subword tokens, then convert to bytes
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
        subword_ids = hf_tokenizer.encode(text, add_special_tokens=True)

        # Convert subword tokens to byte sequences
        byte_ids = []
        for token_id in subword_ids:
            byte_seq = byte_tokenizer.byte_sequences.get(token_id, [])
            byte_ids.extend(byte_seq)

        return np.array(byte_ids, dtype=np.uint32)

    elif tokenizer_type == "subword":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_identifier)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return np.array(token_ids, dtype=np.uint32)

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def download_and_prepare_data(
    output_dir: Path,
    data_fraction: float = 0.01,
    tokenizer_type: str = "byte",
    tokenizer_identifier: str = "allenai/dolma2-tokenizer",
    max_tokens_per_file: int = 100_000_000,
    seed: int = 42,
    streaming: bool = True
) -> List[Path]:
    """
    Download bolmo_mix dataset and prepare .npy files.

    Args:
        output_dir: Directory to save .npy files and data_sources.txt
        data_fraction: Fraction of data to download (0.0 to 1.0)
        tokenizer_type: Either 'byte' or 'subword'
        tokenizer_identifier: HuggingFace tokenizer identifier
        max_tokens_per_file: Maximum number of tokens per .npy file
        seed: Random seed for sampling
        streaming: Whether to use streaming mode (recommended for large datasets)

    Returns:
        List of paths to created .npy files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {data_fraction*100}% of bolmo_mix dataset...")
    log.info(f"Using {tokenizer_type} tokenizer: {tokenizer_identifier}")
    log.info(f"Output directory: {output_dir}")

    # Load dataset
    try:
        if streaming:
            dataset = load_dataset("allenai/bolmo_mix", split="train", streaming=True)
            # For streaming, we can't easily get the total size, so we'll just process
            # until we reach the desired fraction
            log.info("Using streaming mode - processing until reaching target fraction")
        else:
            dataset = load_dataset("allenai/bolmo_mix", split="train")
            log.info(f"Total examples in dataset: {len(dataset)}")

            # Sample a fraction of the data
            if data_fraction < 1.0:
                import random
                random.seed(seed)
                num_samples = int(len(dataset) * data_fraction)
                indices = random.sample(range(len(dataset)), num_samples)
                dataset = dataset.select(indices)
                log.info(f"Selected {num_samples} examples ({data_fraction*100}%)")

    except Exception as e:
        log.error(f"Failed to load dataset: {e}")
        log.info("Make sure you have internet connection and the datasets library installed")
        raise

    # Process and save data
    file_paths = []
    current_tokens = []
    file_idx = 0
    total_tokens = 0
    processed_examples = 0

    # For streaming mode, we need to estimate when to stop
    max_examples = None
    if streaming and data_fraction < 1.0:
        # Rough estimate: bolmo_mix has ~172B tokens, assume avg 1000 tokens per example
        estimated_total_examples = 172_000_000_000 // 1000
        max_examples = int(estimated_total_examples * data_fraction)
        log.info(f"Will process approximately {max_examples} examples")

    try:
        for example in tqdm(dataset, desc="Processing examples", total=max_examples):
            if max_examples and processed_examples >= max_examples:
                break

            # Get text from example
            # The exact field name might vary - adjust as needed
            text = example.get('text', '') or example.get('content', '') or str(example)

            if not text:
                continue

            # Tokenize
            try:
                tokens = tokenize_text(text, tokenizer_type, tokenizer_identifier)
                current_tokens.extend(tokens.tolist())
                total_tokens += len(tokens)
                processed_examples += 1

                # Save file when reaching max tokens per file
                if len(current_tokens) >= max_tokens_per_file:
                    file_path = output_dir / f"bolmo_data_{file_idx:05d}.npy"
                    token_array = np.array(current_tokens, dtype=np.uint32)
                    write_array_to_disk(token_array, file_path)
                    file_paths.append(file_path)
                    log.info(f"Saved {file_path.name} with {len(current_tokens):,} tokens")

                    current_tokens = []
                    file_idx += 1

            except Exception as e:
                log.warning(f"Failed to tokenize example {processed_examples}: {e}")
                continue

        # Save remaining tokens
        if current_tokens:
            file_path = output_dir / f"bolmo_data_{file_idx:05d}.npy"
            token_array = np.array(current_tokens, dtype=np.uint32)
            write_array_to_disk(token_array, file_path)
            file_paths.append(file_path)
            log.info(f"Saved {file_path.name} with {len(current_tokens):,} tokens")

    except KeyboardInterrupt:
        log.info("Interrupted by user. Saving current progress...")
        if current_tokens:
            file_path = output_dir / f"bolmo_data_{file_idx:05d}.npy"
            token_array = np.array(current_tokens, dtype=np.uint32)
            write_array_to_disk(token_array, file_path)
            file_paths.append(file_path)
            log.info(f"Saved {file_path.name} with {len(current_tokens):,} tokens")

    log.info(f"\nProcessing complete!")
    log.info(f"Processed {processed_examples:,} examples")
    log.info(f"Total tokens: {total_tokens:,}")
    log.info(f"Created {len(file_paths)} .npy files")

    return file_paths


def create_data_sources_file(output_dir: Path, file_paths: List[Path]):
    """Create data_sources.txt file listing all .npy files."""
    sources_file = output_dir / "data_sources.txt"

    with open(sources_file, 'w') as f:
        for path in file_paths:
            # Write absolute path
            f.write(f"{path.absolute()}\n")

    log.info(f"Created {sources_file}")
    log.info(f"\nYou can now use this data by setting:")
    log.info(f"  DATA_SOURCE={sources_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare bolmo_mix dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save .npy files and data_sources.txt"
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=0.01,
        help="Fraction of data to download (0.0 to 1.0). Default: 0.01 (1%%)"
    )

    parser.add_argument(
        "--tokenizer",
        choices=["byte", "subword"],
        default="byte",
        help="Tokenizer type to use. Default: byte"
    )

    parser.add_argument(
        "--tokenizer-identifier",
        default="allenai/dolma2-tokenizer",
        help="HuggingFace tokenizer identifier. Default: allenai/dolma2-tokenizer"
    )

    parser.add_argument(
        "--max-tokens-per-file",
        type=int,
        default=100_000_000,
        help="Maximum tokens per .npy file. Default: 100M"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42"
    )

    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (loads entire dataset into memory)"
    )

    args = parser.parse_args()

    # Validate data_fraction
    if not 0.0 < args.data_fraction <= 1.0:
        parser.error("--data-fraction must be between 0.0 and 1.0")

    # Download and prepare data
    file_paths = download_and_prepare_data(
        output_dir=args.output_dir,
        data_fraction=args.data_fraction,
        tokenizer_type=args.tokenizer,
        tokenizer_identifier=args.tokenizer_identifier,
        max_tokens_per_file=args.max_tokens_per_file,
        seed=args.seed,
        streaming=not args.no_streaming
    )

    # Create data_sources.txt
    if file_paths:
        create_data_sources_file(args.output_dir, file_paths)
    else:
        log.error("No files were created!")
        sys.exit(1)


if __name__ == "__main__":
    main()
