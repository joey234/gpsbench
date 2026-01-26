#!/usr/bin/env python3
"""
Single-Task Fine-Tuning Script
Task: Coordinate → Location (Place Association)

Benefits:
- No catastrophic forgetting (single task)
- No task interference
- Can use higher learning rate safely
- Simpler, more focused training
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def load_data(data_path: Path):
    """Load training data"""
    with open(data_path) as f:
        data = json.load(f)
    return data


def format_example(example, tokenizer):
    """Format example for chat template"""
    messages = [
        {"role": "system", "content": example.get('system_prompt', 'You are a helpful assistant.')},
        {"role": "user", "content": example['input']},
        {"role": "assistant", "content": example['output']}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description='Single-task fine-tuning for coordinate → location')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                        help='Base model to fine-tune')
    parser.add_argument('--train-file', type=str, default='data/train_single_task.json',
                        help='Training data file')
    parser.add_argument('--dev-file', type=str, default='data/dev_single_task.json',
                        help='Development data file')
    parser.add_argument('--output-dir', type=str, default='models/qwen3-4b-place-association-lora',
                        help='Output directory for model')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs (3 for large datasets, 10 for small)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (higher is safe for single task)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--auto-adjust', action='store_true',
                        help='Auto-adjust epochs based on dataset size')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (lower = less forgetting, recommended: 8-16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (typically 2x lora-r)')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout (higher = less overfitting, recommended: 0.1-0.2)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--early-stopping-patience', type=int, default=2,
                        help='Number of evaluations with no improvement before stopping (default: 2)')
    parser.add_argument('--early-stopping-threshold', type=float, default=0.001,
                        help='Minimum change to qualify as improvement (default: 0.001)')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping')

    args = parser.parse_args()

    print("=" * 80)
    print("Single-Task Fine-Tuning: Coordinate → Location")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print(f"Task: Place Association (Reverse Geocoding)")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LoRA Rank: {args.lora_r}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    if not args.no_early_stopping:
        print(f"Early Stopping: Enabled (patience={args.early_stopping_patience}, threshold={args.early_stopping_threshold})")
    else:
        print("Early Stopping: Disabled")
    print()

    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if not set (required for batch training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Configure LoRA
    print("Configuring LoRA...")
    print(f"  LoRA rank: {args.lora_r} (lower rank = less capacity = less forgetting)")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  LoRA dropout: {args.lora_dropout} (regularization against overfitting)")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Ensure model is in training mode
    model.train()

    # Load and prepare data
    print("\nLoading training data...")
    train_data = load_data(Path(args.train_file))
    dev_data = load_data(Path(args.dev_file))

    print(f"Train examples: {len(train_data)}")
    print(f"Dev examples: {len(dev_data)}")

    # Auto-adjust epochs based on dataset size
    if args.auto_adjust:
        if len(train_data) > 100000:
            recommended_epochs = 3
        elif len(train_data) > 50000:
            recommended_epochs = 4
        elif len(train_data) > 10000:
            recommended_epochs = 6
        else:
            recommended_epochs = 10

        if recommended_epochs != args.num_epochs:
            print(f"\nAuto-adjusting epochs: {args.num_epochs} → {recommended_epochs}")
            print(f"  Reason: Dataset size {len(train_data)} examples")
            args.num_epochs = recommended_epochs
    else:
        # Warn if using wrong epochs for dataset size
        if len(train_data) > 100000 and args.num_epochs > 5:
            print(f"\n⚠️  WARNING: Large dataset ({len(train_data)} examples) with many epochs ({args.num_epochs})")
            print(f"   Recommended: 3-4 epochs for 100K+ examples")
            print(f"   Consider: --auto-adjust flag or --num-epochs 3")

    # Format examples
    print("\nFormatting examples...")
    train_formatted = [format_example(ex, tokenizer) for ex in train_data]
    dev_formatted = [format_example(ex, tokenizer) for ex in dev_data]

    train_dataset = Dataset.from_list(train_formatted)
    dev_dataset = Dataset.from_list(dev_formatted)

    # Tokenize
    print("Tokenizing...")
    def tokenize_function(examples):
        # Tokenize without padding (collator will handle it)
        # DataCollatorForLanguageModeling will automatically create labels from input_ids
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )
        return result

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"

    # Calculate warmup ratio (5% of total steps is good for preventing forgetting)
    warmup_ratio = 0.05

    print(f"\n🛡️  Anti-Catastrophic Forgetting Safeguards:")
    print(f"  ✓ LoRA (only {args.lora_r} rank adapters trained, base model frozen)")
    print(f"  ✓ Weight decay: 0.01 (L2 regularization)")
    print(f"  ✓ LoRA dropout: {args.lora_dropout} (prevents overfitting)")
    print(f"  ✓ Warmup ratio: {warmup_ratio:.1%} (gradual learning start)")
    print(f"  ✓ LR scheduler: cosine (smooth decay, gentle updates)")
    print(f"  ✓ Early stopping: patience={args.early_stopping_patience} (prevents overtraining)")
    print(f"  ✓ Gradient checkpointing: disabled (LoRA is already memory-efficient)")
    print()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,  # L2 regularization to prevent large weight changes
        warmup_ratio=warmup_ratio,  # Gradual warmup prevents sudden forgetting
        lr_scheduler_type="cosine",  # Smooth decay helps preserve base model
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=False,  # Disabled for LoRA compatibility
        gradient_accumulation_steps=2,
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for GPU
    )

    # Early stopping callback
    callbacks = []
    if not args.no_early_stopping:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )
        callbacks.append(early_stopping)
        print(f"\n✓ Early stopping enabled:")
        print(f"  - Will stop if validation loss doesn't improve for {args.early_stopping_patience} evaluations")
        print(f"  - Minimum improvement threshold: {args.early_stopping_threshold}")
        print(f"  - Best model will be automatically loaded at the end")
        print()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_output = Path(args.output_dir)
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {final_output}")
    print("\nNext steps:")
    print("1. Evaluate on test set:")
    print(f"   python scripts/evaluate_single_task.py --lora-path {final_output}")
    print("2. Test inference:")
    print(f"   python scripts/test_inference.py --lora-path {final_output}")


if __name__ == "__main__":
    main()
