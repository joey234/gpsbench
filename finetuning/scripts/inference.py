"""
Inference script for fine-tuned coordinate-to-location model.

This script loads the LoRA fine-tuned model and performs inference
on coordinate queries.
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str, lora_path: str, device: str = "auto"):
    """
    Load base model and LoRA adapter.

    Args:
        base_model_name: Base model name (e.g., Qwen/Qwen2.5-0.5B)
        lora_path: Path to LoRA adapter
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, tokenizer


def format_prompt(latitude: float, longitude: float) -> str:
    """
    Format coordinate query into model prompt.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Formatted prompt
    """
    system_prompt = "You are a helpful assistant that identifies locations from GPS coordinates."
    instruction = f"What location is at coordinates ({latitude:.6f}, {longitude:.6f})?"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return prompt


def predict_location(
    model,
    tokenizer,
    latitude: float,
    longitude: float,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> str:
    """
    Predict location from coordinates.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Predicted location string
    """
    prompt = format_prompt(latitude, longitude)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract assistant response
    if "<|im_start|>assistant\n" in generated_text:
        response = generated_text.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0].strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

    return response


def interactive_mode(model, tokenizer):
    """
    Interactive inference mode.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
    """
    print("\n=== Interactive Location Prediction ===")
    print("Enter coordinates in format: latitude,longitude")
    print("Example: 40.7128,-74.0060")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("Enter coordinates: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Parse coordinates
            parts = user_input.split(',')
            if len(parts) != 2:
                print("Invalid format. Please use: latitude,longitude")
                continue

            lat = float(parts[0].strip())
            lon = float(parts[1].strip())

            # Validate ranges
            if not (-90 <= lat <= 90):
                print("Latitude must be between -90 and 90")
                continue
            if not (-180 <= lon <= 180):
                print("Longitude must be between -180 and 180")
                continue

            # Predict
            print(f"\nPredicting location for ({lat:.6f}, {lon:.6f})...")
            prediction = predict_location(model, tokenizer, lat, lon)
            print(f"Predicted location: {prediction}\n")

        except ValueError:
            print("Invalid coordinates. Please enter numbers.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(
    model,
    tokenizer,
    coordinates_file: str,
    output_file: str = None
):
    """
    Perform batch inference on a file of coordinates.

    Expected format (CSV): latitude,longitude,ground_truth_location

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        coordinates_file: Path to CSV file with coordinates
        output_file: Optional output file for predictions
    """
    import csv

    results = []

    with open(coordinates_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        print(f"\nProcessing coordinates from {coordinates_file}...")

        for i, row in enumerate(reader):
            if len(row) < 2:
                continue

            lat = float(row[0])
            lon = float(row[1])
            ground_truth = row[2] if len(row) > 2 else "Unknown"

            prediction = predict_location(model, tokenizer, lat, lon)

            results.append({
                'latitude': lat,
                'longitude': lon,
                'ground_truth': ground_truth,
                'prediction': prediction,
            })

            print(f"{i+1}. ({lat:.6f}, {lon:.6f}) -> {prediction}")

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['latitude', 'longitude', 'ground_truth', 'prediction']
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned location prediction model"
    )

    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Base model name'
    )
    parser.add_argument(
        '--lora-path',
        type=str,
        default='../models/qwen-lora-location',
        help='Path to LoRA adapter'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'batch', 'single'],
        default='interactive',
        help='Inference mode'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help='Latitude (for single mode)'
    )
    parser.add_argument(
        '--lon',
        type=float,
        help='Longitude (for single mode)'
    )
    parser.add_argument(
        '--batch-file',
        type=str,
        help='CSV file with coordinates (for batch mode)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for batch predictions'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (0 = greedy)'
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.base_model, args.lora_path)

    # Run inference based on mode
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer)

    elif args.mode == 'single':
        if args.lat is None or args.lon is None:
            print("Error: --lat and --lon required for single mode")
            return

        prediction = predict_location(
            model,
            tokenizer,
            args.lat,
            args.lon,
            temperature=args.temperature
        )
        print(f"\nCoordinates: ({args.lat:.6f}, {args.lon:.6f})")
        print(f"Predicted location: {prediction}")

    elif args.mode == 'batch':
        if args.batch_file is None:
            print("Error: --batch-file required for batch mode")
            return

        batch_inference(
            model,
            tokenizer,
            args.batch_file,
            args.output_file
        )


if __name__ == '__main__':
    main()
