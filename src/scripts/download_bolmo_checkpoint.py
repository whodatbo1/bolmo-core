#!/usr/bin/env python3
"""
Download Bolmo checkpoint from HuggingFace for training/fine-tuning.

The Bolmo checkpoints on HuggingFace include both:
1. HuggingFace format (in root directory)
2. OLMo-core format (in olmo_core/ subdirectory)

This script downloads the olmo-core format which can be directly loaded
by bolmo-core without any conversion needed.

Usage:
    python src/scripts/download_bolmo_checkpoint.py --model 1b --output-dir /path/to/output
    python src/scripts/download_bolmo_checkpoint.py --model 7b --output-dir /path/to/output
"""

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


BOLMO_MODELS = {
    "1b": "allenai/Bolmo-1B",
    "7b": "allenai/Bolmo-7B",
}


def download_bolmo_checkpoint(model_size: str, output_dir: Path, revision: str = "main") -> None:
    """
    Download Bolmo checkpoint from HuggingFace.

    Args:
        model_size: Size of the model ("1b" or "7b")
        output_dir: Directory where checkpoint will be saved
        revision: HuggingFace revision/branch to download from
    """
    if model_size not in BOLMO_MODELS:
        raise ValueError(f"Invalid model size '{model_size}'. Must be one of: {list(BOLMO_MODELS.keys())}")

    hf_model = BOLMO_MODELS[model_size]
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading Bolmo {model_size.upper()} checkpoint")
    log.info(f"HuggingFace Model: {hf_model}")
    log.info(f"Revision: {revision}")
    log.info(f"Output Directory: {output_dir}")
    log.info("")
    log.info("Note: The Bolmo HF checkpoints include a pre-converted olmo-core format")
    log.info("      in the 'olmo_core/' subdirectory, so no conversion is needed.")
    log.info("")

    # Download the olmo_core subdirectory from the HF repo
    # This contains the model_and_optim directory and config.json in olmo-core format
    temp_dir = output_dir / "temp_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Downloading olmo-core checkpoint from HuggingFace...")
        snapshot_download(
            repo_id=hf_model,
            revision=revision,
            allow_patterns=["olmo_core/*"],
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )

        # Move files from olmo_core subdirectory to root of output directory
        olmo_core_dir = temp_dir / "olmo_core"
        if olmo_core_dir.exists():
            log.info("Extracting olmo-core checkpoint...")
            for item in olmo_core_dir.iterdir():
                dest = output_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
        else:
            raise RuntimeError(
                f"Expected 'olmo_core/' subdirectory not found in {hf_model}. "
                "The checkpoint structure may have changed."
            )

        log.info("")
        log.info("✓ Checkpoint downloaded successfully!")
        log.info("")
        log.info(f"The checkpoint is now available at: {output_dir}")
        log.info(f"- Model weights: {output_dir}/model_and_optim/")
        log.info(f"- Config: {output_dir}/config.json")
        log.info("")
        log.info("To use this checkpoint in training, set:")
        log.info(f"  BOLMO_CKPT_PATH={output_dir}/model_and_optim")

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def verify_checkpoint(output_dir: Path) -> bool:
    """
    Verify that the downloaded checkpoint has the expected structure.

    Args:
        output_dir: Directory containing the checkpoint

    Returns:
        True if checkpoint structure is valid
    """
    required_files = [
        output_dir / "config.json",
        output_dir / "model_and_optim",
    ]

    for required_file in required_files:
        if not required_file.exists():
            log.error(f"Missing required file/directory: {required_file}")
            return False

    log.info("Checkpoint structure verified successfully")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=list(BOLMO_MODELS.keys()),
        help="Size of the Bolmo model to download",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoint will be saved",
    )
    parser.add_argument(
        "-r",
        "--revision",
        type=str,
        default="main",
        help="HuggingFace revision/branch to download from (default: main)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify checkpoint structure after download",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        download_bolmo_checkpoint(
            model_size=args.model,
            output_dir=args.output_dir,
            revision=args.revision,
        )

        if args.verify:
            log.info("")
            log.info("Verifying checkpoint structure...")
            if not verify_checkpoint(args.output_dir):
                log.error("Checkpoint verification failed!")
                return 1

        return 0

    except Exception as e:
        log.error(f"Failed to download checkpoint: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
