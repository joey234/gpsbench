"""
Tinker-based training script for GPSBench fine-tuning.

Uses tinker-cookbook's supervised learning interface for LoRA fine-tuning.

Example usage:
    # Example run (quick test)
    python train.py --max-samples 100 --max-steps 50

    # Full training
    python train.py

    # Custom configuration
    python train.py --learning-rate 3e-4 --batch-size 128 --lora-rank 128
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import chz
import datasets as hf_datasets
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
)

# Import GPSBench dataset builders
from gpsbench_datasets import GPSBenchInstructionDataset, GPSBenchSplitDataset

# Import GPSBench evaluators
from gpsbench_evaluators import create_gpsbench_evaluator

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")

    # Verify Tinker API key
    tinker_key = os.getenv('TINKER_API_KEY')
    if not tinker_key or tinker_key == 'your_tinker_api_key_here':
        print("\n⚠ WARNING: TINKER_API_KEY not set!")
        print("  Get your key from: https://tinker-docs.thinkingmachines.ai/")
        print("  Run './setup_api_keys.sh' to configure")
        raise ValueError("TINKER_API_KEY not configured")
    else:
        print(f"✓ Tinker API key found ({len(tinker_key)} chars)")

except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")


# Dataset builders are now in gpsbench_datasets.py


@chz.chz
class TrainingConfig:
    """Training configuration."""

    # Model and data
    # Tinker supported models: Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-8B-Base,
    # Qwen/Qwen3-30B-A3B-Base, Qwen/Qwen3-30B-A3B-Instruct-2507, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.1-8B
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 30B MoE instruct model

    # Dataset loading mode
    dataset_mode: str = "instruction_files"  # "instruction_files" or "split_files"
    data_path: str = "../data"  # For instruction_files mode
    gpsbench_root: str = "../../"  # For split_files mode

    # Custom data files (optional, for augmented data)
    train_file: str | None = None  # Custom train file path
    dev_file: str | None = None    # Custom dev file path

    # Data filtering (for both modes)
    track: str = "both"  # "pure_gps", "applied", or "both" (for split_files mode)
    tasks: List[str] | None = None  # Filter specific tasks (e.g., ["task1_format_conversion"])
    filter_tasks: List[str] | None = None  # Filter by task name (for instruction_files mode)
    include_reasoning: bool = False  # Include chain-of-thought reasoning in training outputs

    # Training hyperparameters
    # NOTE: Lower LR (2e-5) prevents catastrophic forgetting of pre-trained knowledge
    learning_rate: float = 2e-5  # Changed from 5e-4 - much lower to preserve model capabilities
    lr_schedule: str = "linear"  # Linear decay for smooth convergence (cosine may not be supported)
    num_epochs: int = 1  # Changed from 3 - single epoch to prevent overfitting
    batch_size: int = 32  # Reduced from 64 for memory safety with 30B model

    # LoRA config
    lora_rank: int = 64

    # Dataset limits
    max_samples: int | None = None
    max_eval_samples: int | None = None
    max_length: int = 16384

    # Checkpointing and evaluation
    # More frequent evaluation to catch overfitting early
    save_every: int = 200
    eval_every: int = 100  # Changed from 500 - evaluate more frequently
    infrequent_eval_every: int = 500

    # Logging
    wandb_project: str | None = "gpsbench-finetuning"
    wandb_name: str | None = None
    log_path: str | None = None

    # Infrastructure
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    # Evaluation
    enable_task_eval: bool = True  # Enable GPSBench task-specific evaluation
    eval_max_samples: int | None = None  # Max samples for task evaluation (None = use full dev set)

    # Advanced
    max_steps: int | None = None
    renderer_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"  # "ask", "resume", "delete", or "raise"


def cli_main(config: TrainingConfig):
    """Main training function."""

    print("=" * 60)
    print("GPSBench Fine-tuning with Tinker")
    print("=" * 60)
    print(f"\nModel: {config.model_name}")
    print(f"Data path: {config.data_path}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Max samples: {config.max_samples or 'all'}")
    print(f"Max steps: {config.max_steps or 'auto'}")
    print()

    # Create log directory
    if config.log_path is not None:
        log_path = config.log_path
    else:
        model_name = config.model_name.replace("/", "-")
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = f"gpsbench-{model_name}-{config.lora_rank}rank-{config.learning_rate}lr-{config.batch_size}batch-{date_time}"
        log_path = f"./outputs/{run_name}"

    if config.wandb_name is not None:
        wandb_name = config.wandb_name
    else:
        wandb_name = Path(log_path).name

    print(f"Log path: {log_path}")
    print(f"W&B name: {wandb_name}")
    print()

    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    # Get recommended renderer for model
    if config.renderer_name:
        renderer_name = config.renderer_name
    else:
        # Auto-detect renderer based on model name
        try:
            renderer_name = model_info.get_recommended_renderer_name(config.model_name)
            print(f"Using auto-detected renderer: {renderer_name}")
        except Exception as e:
            # Fallback based on model family
            if "llama" in config.model_name.lower():
                renderer_name = "llama3"
            elif "qwen" in config.model_name.lower():
                renderer_name = "qwen_chat"
            elif "deepseek" in config.model_name.lower():
                renderer_name = "deepseek_chat"
            else:
                renderer_name = "llama3"  # Safe default
            print(f"⚠ Could not auto-detect renderer, using fallback: {renderer_name}")

    # Create dataset builder based on mode
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )

    if config.dataset_mode == "split_files":
        print(f"Loading from split files (track: {config.track})")
        if config.tasks:
            print(f"Filtering tasks: {config.tasks}")

        dataset_builder = GPSBenchSplitDataset(
            common_config=common_config,
            gpsbench_root=config.gpsbench_root,
            track=config.track,
            tasks=config.tasks,
            max_samples=config.max_samples,
            max_eval_samples=config.max_eval_samples,
            include_reasoning=config.include_reasoning,
        )
    else:  # instruction_files mode (default)
        print(f"Loading from instruction files: {config.data_path}")
        if config.filter_tasks:
            print(f"Filtering tasks: {config.filter_tasks}")
        if config.train_file:
            print(f"Using custom train file: {config.train_file}")
        if config.dev_file:
            print(f"Using custom dev file: {config.dev_file}")

        dataset_builder = GPSBenchInstructionDataset(
            common_config=common_config,
            data_path=config.data_path,
            train_file=config.train_file,
            dev_file=config.dev_file,
            max_samples=config.max_samples,
            max_eval_samples=config.max_eval_samples,
            filter_tasks=config.filter_tasks,
        )

    # Calculate num_epochs or use max_steps workaround
    # Note: tinker-cookbook doesn't support max_steps directly
    # If max_steps is set, we use a very small num_epochs and rely on manual stopping
    if config.max_steps is not None:
        print(f"⚠ Warning: max_steps is set to {config.max_steps}")
        print("  tinker-cookbook doesn't support max_steps - training will run for epochs")
        print("  You'll need to manually stop training at the desired step")
        # Use num_epochs but user should monitor
        effective_num_epochs = config.num_epochs
    else:
        effective_num_epochs = config.num_epochs

    # Create GPSBench task evaluator (optional)
    evaluator_builders = []
    if config.enable_task_eval:
        print(f"Setting up GPSBench task evaluator (max {config.eval_max_samples} samples)...")
        gpsbench_evaluator = create_gpsbench_evaluator(
            eval_dataset_builder=dataset_builder,
            model_name=config.model_name,
            renderer_name=renderer_name,
            name="gpsbench_eval",
            max_samples=config.eval_max_samples
        )
        evaluator_builders.append(gpsbench_evaluator)
    else:
        print("Task-specific evaluation disabled (set enable_task_eval=True to enable)")

    # Create training config
    train_config = train.Config(
        log_path=log_path,
        model_name=config.model_name,
        load_checkpoint_path=config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=evaluator_builders,
        infrequent_evaluator_builders=[],
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=effective_num_epochs,
        base_url=config.base_url,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=config.lora_rank,
        save_every=config.save_every,
        eval_every=config.eval_every,
        infrequent_eval_every=config.infrequent_eval_every,
    )

    # Run training
    print("=" * 60)
    print("Starting training on Tinker...")
    print("=" * 60)
    print()

    asyncio.run(train.main(train_config))

    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nModel saved to: {log_path}")


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
