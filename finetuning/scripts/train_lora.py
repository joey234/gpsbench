"""
LoRA Fine-tuning script for coordinate-to-location prediction.

This script fine-tunes Qwen/Qwen3-0.6B using LoRA (Low-Rank Adaptation)
for robust location prediction from noisy GPS coordinates.
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, List
import argparse
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training."""

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as it's available
    use_4bit: bool = True  # Use 4-bit quantization to save memory

    # LoRA settings
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 256

    # Paths
    data_dir: str = "../data"
    output_dir: str = "../models/qwen-lora-location"
    checkpoint_dir: str = "../models/checkpoints"


def load_dataset_from_json(json_file: str) -> Dataset:
    """Load dataset from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter to only keep fields needed for training
    # This avoids PyArrow issues with nested metadata dictionaries
    # Convert all fields to strings to avoid type errors
    filtered_data = []
    for example in data:
        # Get values and ensure they're strings
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        # Convert to string if not already
        if not isinstance(instruction, str):
            instruction = str(instruction)
        if not isinstance(input_text, str):
            input_text = str(input_text)
        if not isinstance(output, str):
            output = json.dumps(output) if isinstance(output, (dict, list)) else str(output)

        filtered_example = {
            'instruction': instruction,
            'input': input_text,
            'output': output,
        }
        filtered_data.append(filtered_example)

    return Dataset.from_list(filtered_data)


def format_instruction(example: Dict) -> str:
    """
    Format example into instruction-following format with task-specific system prompt.

    Template:
    <|im_start|>system
    {task-specific system prompt}<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    """
    # Import task prompt mapping
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from task_prompt_mapping import get_system_prompt_for_task

    # Get task-specific system prompt
    task_name = None
    if 'metadata' in example and 'task' in example['metadata']:
        task_name = example['metadata']['task']

    system_prompt = get_system_prompt_for_task(task_name) if task_name else \
                    "You are a helpful assistant that identifies locations from GPS coordinates."

    formatted = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )

    return formatted


def preprocess_function(examples: Dict, tokenizer, max_length: int = 256) -> Dict:
    """
    Preprocess examples for training.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized inputs
    """
    # Format instructions
    texts = [format_instruction(ex) for ex in examples]

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    # Mask padding tokens in labels
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100

    return model_inputs


def prepare_model_and_tokenizer(config: LoRATrainingConfig):
    """
    Load and prepare model and tokenizer with LoRA.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",  # Required for decoder-only models
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    if config.use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: LoRATrainingConfig):
    """
    Main training function.

    Args:
        config: Training configuration
    """
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_file = Path(config.data_dir) / "train_dataset.json"
    val_file = Path(config.data_dir) / "val_dataset.json"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_file}\n"
            "Please run generate_dataset.py first!"
        )

    train_dataset = load_dataset_from_json(str(train_file))
    val_dataset = load_dataset_from_json(str(val_file))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)

    # Preprocess datasets
    print("Preprocessing datasets...")

    def tokenize(examples):
        # Handle batch processing
        formatted_examples = []
        for i in range(len(examples['instruction'])):
            formatted_examples.append({
                'instruction': examples['instruction'][i],
                'input': examples['input'][i],
                'output': examples['output'][i],
            })

        texts = [format_instruction(ex) for ex in formatted_examples]

        model_inputs = tokenizer(
            texts,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )

        # Set labels
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        # Mask padding tokens
        for i, label in enumerate(model_inputs["labels"]):
            model_inputs["labels"][i] = [
                -100 if token_id == tokenizer.pad_token_id else token_id
                for token_id in label
            ]

        return model_inputs

    tokenized_train = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )

    tokenized_val = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.checkpoint_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen model with LoRA for location prediction"
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help='Model name or path'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing train_dataset.json and val_dataset.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,  # Will be auto-generated based on model name if not provided
        help='Output directory for fine-tuned model (default: auto-generated from model name)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--no-4bit',
        action='store_true',
        help='Disable 4-bit quantization'
    )

    args = parser.parse_args()

    # Auto-generate output directory based on model name if not provided
    if args.output_dir is None:
        # Convert model name to directory-safe name
        # E.g., "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b"
        model_slug = args.model_name.replace('/', '-').replace('_', '-').lower()
        # Remove common prefixes for cleaner names
        model_slug = model_slug.replace('qwen-', '')
        args.output_dir = f'../models/{model_slug}-lora'
        print(f"Output directory not specified, using: {args.output_dir}")

    # Create config
    config = LoRATrainingConfig(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.no_4bit,
    )

    # Run training
    train(config)


if __name__ == '__main__':
    main()
