#!/usr/bin/env python3
"""
Compare Base Model (Zero-Shot) vs Fine-Tuned Model on GPSBench

This script evaluates both the base model and fine-tuned LoRA model
on the full GPSBench benchmark and generates a comparison report.

Supports both standard transformers and vLLM for faster inference.

Usage:
    # Compare on both tracks with vLLM (FAST - recommended)
    python compare_base_vs_finetuned.py \
        --base-model Qwen/Qwen2.5-0.5B \
        --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
        --track both

    # Compare with multiple GPUs (vLLM)
    python compare_base_vs_finetuned.py \
        --base-model Qwen/Qwen2.5-7B \
        --lora-path ../finetuning/models/qwen2.5-7b-lora \
        --track both \
        --tensor-parallel-size 2

    # Compare without vLLM (slower, less memory)
    python compare_base_vs_finetuned.py \
        --base-model Qwen/Qwen2.5-0.5B \
        --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
        --track both \
        --no-vllm

    # Reduce memory usage (if you get KV cache errors)
    python compare_base_vs_finetuned.py \
        --base-model Qwen/Qwen2.5-0.5B \
        --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
        --track both \
        --gpu-memory-utilization 0.80 \
        --max-model-len 1024

    # Quick test (10 samples per task)
    python compare_base_vs_finetuned.py \
        --base-model Qwen/Qwen2.5-0.5B \
        --lora-path ../finetuning/models/qwen2.5-0.5b-lora \
        --track both \
        --max-samples 10
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import torch
from tqdm import tqdm

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import LLMClient
from track_config import get_track_info

# Import track evaluators
try:
    from evaluate_pure_gps import PureGPSEvaluator
    from evaluate_applied import AppliedGPSEvaluator
except ImportError as e:
    print(f"Error: Could not import track evaluators: {e}")
    print("\nMake sure you're running from the evaluation/ directory")
    print("and that evaluate_pure_gps.py and evaluate_applied.py exist.")
    sys.exit(1)

# Import fine-tuning inference
sys.path.insert(0, str(Path(__file__).parent.parent / "finetuning" / "scripts"))
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    print("Error: transformers and peft are required. Install with:")
    print("  pip install transformers peft")
    sys.exit(1)

# Try to import vLLM for faster inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Using standard transformers (slower).")
    print("For faster inference, install vLLM: pip install vllm")


def remove_think_tags(text: str) -> str:
    """
    Remove <think> tags and their content from model outputs.

    Qwen models may output reasoning in <think>...</think> tags from pre-training.
    We want to remove this to get only the final answer.

    Args:
        text: Model output text

    Returns:
        Text with <think> tags and content removed
    """
    import re
    # Remove <think>...</think> blocks (non-greedy)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any remaining <think> or </think> tags
    cleaned = re.sub(r'</?think>', '', cleaned, flags=re.IGNORECASE)
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned).strip()
    return cleaned


class LocalModelClient:
    """
    Adapter to make local models (base and fine-tuned) compatible with LLMClient interface.
    """

    def __init__(self, model_name: str, lora_path: str = None, device: str = "auto"):
        """
        Initialize local model client.

        Args:
            model_name: Base model name (e.g., Qwen/Qwen2.5-0.5B)
            lora_path: Path to LoRA adapter (None for base model)
            device: Device to load model on
        """
        self.model_name = model_name
        self.lora_path = lora_path
        self.is_finetuned = lora_path is not None

        print(f"Loading {'fine-tuned' if self.is_finetuned else 'base'} model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        print("Model loaded successfully!")

    def query(self, prompt: str, system_prompt: str = None, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """
        Query the model with a prompt.

        Args:
            prompt: Input prompt
            system_prompt: System prompt (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        # Use provided system prompt or default to match training
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that identifies locations from GPS coordinates."

        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        if "<|im_start|>assistant\n" in generated_text:
            response = generated_text.split("<|im_start|>assistant\n")[-1]
            response = response.split("<|im_end|>")[0].strip()
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(formatted_prompt):].strip()

        # Remove <think> tags from Qwen base model pre-training
        response = remove_think_tags(response)

        return response

    def get_model_name(self) -> str:
        """Get model identifier"""
        if self.is_finetuned:
            return f"{self.model_name}-finetuned"
        return f"{self.model_name}-base"


class VLLMModelClient:
    """
    vLLM-based client for faster batched inference (10-20x speedup).
    Supports both base models and LoRA adapters.
    """

    def __init__(self, model_name: str, lora_path: str = None, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85, max_model_len: int = 2048):
        """
        Initialize vLLM model client.

        Args:
            model_name: Base model name (e.g., Qwen/Qwen2.5-0.5B)
            lora_path: Path to LoRA adapter (None for base model)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0, default: 0.85)
            max_model_len: Maximum sequence length (default: 2048, reduces KV cache memory)
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")

        self.model_name = model_name
        self.lora_path = lora_path
        self.is_finetuned = lora_path is not None

        print(f"Loading {'fine-tuned' if self.is_finetuned else 'base'} model with vLLM: {model_name}")

        # vLLM configuration
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,  # Limit KV cache memory usage
        }

        # Enable LoRA if adapter provided
        if lora_path:
            print(f"Loading LoRA adapter from: {lora_path}")
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64  # Adjust based on your LoRA config

        self.llm = LLM(**llm_kwargs)

        # Load tokenizer for prompt formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print("vLLM model loaded successfully!")

    def query(self, prompt: str, system_prompt: str = None, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """
        Query the model with a single prompt.

        Args:
            prompt: Input prompt
            system_prompt: System prompt (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        return self.query_batch([prompt], system_prompt, temperature, max_tokens)[0]

    def query_batch(self, prompts: List[str], system_prompt: str = None, temperature: float = 0.1, max_tokens: int = 2048) -> List[str]:
        """
        Query the model with a batch of prompts (much faster).

        Args:
            prompts: List of input prompts
            system_prompt: System prompt (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of generated responses
        """
        # Use provided system prompt or default to match training
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that identifies locations from GPS coordinates."

        formatted_prompts = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for prompt in prompts
        ]

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.1,
            max_tokens=max_tokens,
            stop=["<|im_end|>"],
            skip_special_tokens=False,
        )

        # Generate with vLLM (batched)
        if self.is_finetuned:
            # Use LoRA adapter
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("finetuned", 1, self.lora_path)
            outputs = self.llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
        else:
            # Base model
            outputs = self.llm.generate(formatted_prompts, sampling_params)

        # Extract responses
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            # Clean up response (remove special tokens)
            response = generated_text.split("<|im_end|>")[0].strip()
            # Remove <think> tags from Qwen base model pre-training
            response = remove_think_tags(response)
            responses.append(response)

        return responses

    def get_model_name(self) -> str:
        """Get model identifier"""
        if self.is_finetuned:
            return f"{self.model_name}-finetuned-vllm"
        return f"{self.model_name}-base-vllm"


def evaluate_model_on_track(
    model_client,  # Can be LocalModelClient or VLLMModelClient
    track: str,
    data_dir: str = "../data",
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate a model on a specific track.

    Args:
        model_client: Model client instance
        track: Track name ('pure_gps' or 'applied')
        data_dir: Path to data directory
        max_samples: Maximum samples per task (None for all)

    Returns:
        Evaluation results dictionary
    """
    # Convert LocalModelClient to LLMClient-compatible interface
    class ClientWrapper:
        def __init__(self, client):
            self.client = client
            self.provider = "local"
            self.model = client.get_model_name()

        def query(self, prompt, **kwargs):
            return self.client.query(prompt, **kwargs)

        def query_batch(self, prompts, **kwargs):
            """
            Batch query method - exposes underlying client's batch capability.

            Returns list of response strings.
            """
            if hasattr(self.client, 'query_batch'):
                return self.client.query_batch(prompts, **kwargs)
            else:
                # Fall back to sequential if client doesn't support batching
                return [self.client.query(p, **kwargs) for p in prompts]

        def generate(self, prompt: str, system_prompt: str = None, temperature: float = None, max_tokens: int = None):
            """
            Generate response compatible with LLMClient interface.

            Returns a mock LLMResponse object with the generated text.
            """
            # Use defaults if not specified
            temp = temperature if temperature is not None else 0.1
            max_tok = max_tokens if max_tokens is not None else 2048

            # Get response from underlying client
            response_text = self.client.query(prompt, system_prompt=system_prompt, temperature=temp, max_tokens=max_tok)

            # Create mock LLMResponse object
            class MockResponse:
                def __init__(self, text):
                    self.text = text
                    self.model = "local"
                    self.prompt_tokens = 0
                    self.completion_tokens = 0
                    self.total_tokens = 0
                    self.latency_ms = 0

            return MockResponse(response_text)

    wrapped_client = ClientWrapper(model_client)

    # Create appropriate evaluator
    if track == "pure_gps":
        evaluator = PureGPSEvaluator(wrapped_client, data_dir=data_dir)
    elif track == "applied":
        evaluator = AppliedGPSEvaluator(wrapped_client, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown track: {track}")

    # Get track configuration
    track_config = get_track_info(track)

    print(f"\n{'='*80}")
    print(f"Evaluating on {track.upper()} track")
    print(f"Model: {model_client.get_model_name()}")
    print(f"{'='*80}\n")

    # Evaluate all tasks in the track
    all_results = {}
    total_correct = 0
    total_tests = 0

    for tier_name, tier_info in track_config.items():
        if tier_name == "name":  # Skip metadata fields
            continue

        print(f"\n--- {tier_name} ---")

        tier_layer = tier_info.get("layer", "")

        for task_name, task_info in tier_info.get("tasks", {}).items():
            try:
                # Evaluate task
                task_results = evaluator.evaluate_task(
                    layer=tier_layer,
                    task_name=task_name,
                    filename=task_info.get("file", task_info.get("filename", "")),
                    subtasks=task_info.get("subtasks"),
                    max_samples=max_samples
                )

                all_results[task_name] = task_results

                # Update totals (metrics are inside 'metrics' dict)
                if "metrics" in task_results:
                    metrics = task_results["metrics"]
                    total_correct += metrics.get("correct_count", 0)
                    total_tests += metrics.get("total_count", 0)
                    accuracy = metrics.get("accuracy", 0)
                    print(f"✓ {task_name}: {accuracy:.1f}%")
                else:
                    print(f"✓ {task_name}: {task_results.get('accuracy', 0)*100:.1f}%")

            except Exception as e:
                print(f"✗ {task_name}: Error - {e}")
                all_results[task_name] = {"error": str(e)}

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0

    return {
        "track": track,
        "model": model_client.get_model_name(),
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "tasks": all_results,
    }


def compare_models(
    base_model: str,
    lora_path: str,
    track: str = "both",
    data_dir: str = "../data",
    output_dir: str = "results",
    max_samples: int = None,
    use_vllm: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 2048
):
    """
    Compare base model and fine-tuned model on GPSBench.

    Args:
        base_model: Base model name
        lora_path: Path to LoRA adapter
        track: Track to evaluate ('pure_gps', 'applied', or 'both')
        data_dir: Path to data directory
        output_dir: Path to output directory
        max_samples: Maximum samples per task (None for all)
        use_vllm: Use vLLM for faster inference (default: True)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.85)
        max_model_len: Maximum sequence length for vLLM (default: 2048, reduces KV cache)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Determine which tracks to evaluate
    if track == "both":
        tracks = ["pure_gps", "applied"]
    else:
        tracks = [track]

    comparison_results = {
        "base_model": base_model,
        "lora_path": lora_path,
        "timestamp": datetime.now().isoformat(),
        "tracks": {}
    }

    # Choose client type
    if use_vllm and VLLM_AVAILABLE:
        print(f"\n🚀 Using vLLM for faster inference (10-20x speedup)")
        print(f"📊 GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"📏 Max sequence length: {max_model_len}")
        ClientClass = VLLMModelClient
        client_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len
        }
    else:
        if use_vllm and not VLLM_AVAILABLE:
            print(f"\n⚠️  vLLM requested but not available. Falling back to standard transformers.")
        print(f"\n⏱️  Using standard transformers (slower)")
        ClientClass = LocalModelClient
        client_kwargs = {}

    for track_name in tracks:
        print(f"\n{'='*80}")
        print(f"EVALUATING TRACK: {track_name.upper()}")
        print(f"{'='*80}\n")

        # Evaluate base model
        print("\n### BASE MODEL (Zero-Shot) ###\n")
        base_client = ClientClass(base_model, lora_path=None, **client_kwargs)
        base_results = evaluate_model_on_track(
            base_client, track_name, data_dir, max_samples
        )

        # Clean up base model before loading fine-tuned model
        print("\n🧹 Cleaning up base model from GPU memory...")
        del base_client
        gc.collect()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✅ GPU memory cleared. Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

        # Evaluate fine-tuned model
        print("\n### FINE-TUNED MODEL ###\n")
        finetuned_client = ClientClass(base_model, lora_path=lora_path, **client_kwargs)
        finetuned_results = evaluate_model_on_track(
            finetuned_client, track_name, data_dir, max_samples
        )

        # Clean up fine-tuned model after evaluation
        print("\n🧹 Cleaning up fine-tuned model from GPU memory...")
        del finetuned_client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Store results
        comparison_results["tracks"][track_name] = {
            "base": base_results,
            "finetuned": finetuned_results,
        }

        # Print comparison
        print(f"\n{'='*80}")
        print(f"COMPARISON: {track_name.upper()}")
        print(f"{'='*80}")
        print(f"\nBase Model:      {base_results['overall_accuracy']*100:.2f}% "
              f"({base_results['total_correct']}/{base_results['total_tests']})")
        print(f"Fine-Tuned:      {finetuned_results['overall_accuracy']*100:.2f}% "
              f"({finetuned_results['total_correct']}/{finetuned_results['total_tests']})")

        improvement = finetuned_results['overall_accuracy'] - base_results['overall_accuracy']
        print(f"Improvement:     {improvement*100:+.2f}%")
        print(f"{'='*80}\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = base_model.replace("/", "_").replace("-", "_")

    # Save combined comparison file
    output_file = output_path / f"comparison_{model_slug}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\n✅ Combined results saved to: {output_file}")

    # Save separate files for each model
    base_results_only = {
        "model": base_model,
        "model_type": "base (zero-shot)",
        "timestamp": comparison_results["timestamp"],
        "tracks": {}
    }

    finetuned_results_only = {
        "model": base_model,
        "lora_path": lora_path,
        "model_type": "fine-tuned",
        "timestamp": comparison_results["timestamp"],
        "tracks": {}
    }

    # Extract base and finetuned results
    for track_name, track_data in comparison_results["tracks"].items():
        base_results_only["tracks"][track_name] = track_data["base"]
        finetuned_results_only["tracks"][track_name] = track_data["finetuned"]

    # Save base model predictions
    base_output_file = output_path / f"base_{model_slug}_{timestamp}.json"
    with open(base_output_file, 'w') as f:
        json.dump(base_results_only, f, indent=2)
    print(f"✅ Base model predictions saved to: {base_output_file}")

    # Save finetuned model predictions
    finetuned_output_file = output_path / f"finetuned_{model_slug}_{timestamp}.json"
    with open(finetuned_output_file, 'w') as f:
        json.dump(finetuned_results_only, f, indent=2)
    print(f"✅ Fine-tuned model predictions saved to: {finetuned_output_file}")

    # Generate summary report
    generate_summary_report(comparison_results, output_path)


def generate_summary_report(results: Dict, output_dir: Path):
    """Generate a human-readable summary report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"comparison_summary_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GPSBench: Base Model vs Fine-Tuned Model Comparison\n")
        f.write("="*80 + "\n\n")

        f.write(f"Base Model: {results['base_model']}\n")
        f.write(f"LoRA Path: {results['lora_path']}\n")
        f.write(f"Evaluation Date: {results['timestamp']}\n")
        f.write("\n")

        for track_name, track_results in results["tracks"].items():
            f.write("-"*80 + "\n")
            f.write(f"Track: {track_name.upper()}\n")
            f.write("-"*80 + "\n\n")

            base = track_results["base"]
            finetuned = track_results["finetuned"]

            f.write(f"Base Model (Zero-Shot):\n")
            f.write(f"  Overall Accuracy: {base['overall_accuracy']*100:.2f}%\n")
            f.write(f"  Correct: {base['total_correct']} / {base['total_tests']}\n\n")

            f.write(f"Fine-Tuned Model:\n")
            f.write(f"  Overall Accuracy: {finetuned['overall_accuracy']*100:.2f}%\n")
            f.write(f"  Correct: {finetuned['total_correct']} / {finetuned['total_tests']}\n\n")

            improvement = finetuned['overall_accuracy'] - base['overall_accuracy']
            f.write(f"Improvement: {improvement*100:+.2f}%\n")

            # Task-by-task comparison
            f.write("\nTask-by-Task Results:\n\n")
            f.write(f"{'Task':<40} {'Base':<10} {'Fine-Tuned':<10} {'Δ':<10}\n")
            f.write("-"*70 + "\n")

            for task_name in base['tasks'].keys():
                if task_name in finetuned['tasks']:
                    base_acc = base['tasks'][task_name].get('accuracy', 0)
                    ft_acc = finetuned['tasks'][task_name].get('accuracy', 0)
                    delta = ft_acc - base_acc

                    f.write(f"{task_name:<40} {base_acc*100:>6.1f}%    "
                           f"{ft_acc*100:>6.1f}%    {delta*100:>+6.1f}%\n")

            f.write("\n")

    print(f"✅ Summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model and fine-tuned model on GPSBench"
    )

    parser.add_argument(
        '--base-model',
        type=str,
        required=True,
        help='Base model name (e.g., Qwen/Qwen2.5-0.5B)'
    )
    parser.add_argument(
        '--lora-path',
        type=str,
        required=True,
        help='Path to LoRA adapter directory'
    )
    parser.add_argument(
        '--track',
        type=str,
        default='both',
        choices=['pure_gps', 'applied', 'both'],
        help='Track(s) to evaluate on'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Path to GPSBench data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Path to output directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per task (for quick testing)'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        default=True,
        help='Use vLLM for faster inference (default: True)'
    )
    parser.add_argument(
        '--no-vllm',
        action='store_true',
        help='Disable vLLM and use standard transformers'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism (vLLM only)'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.85,
        help='GPU memory utilization fraction for vLLM (default: 0.85). Lower this if you get memory errors.'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=2048,
        help='Maximum sequence length for vLLM (default: 2048). Limits KV cache memory usage. Use 1024-4096 for typical tasks.'
    )

    args = parser.parse_args()

    # Determine whether to use vLLM
    use_vllm = args.use_vllm and not args.no_vllm

    # Run comparison
    compare_models(
        base_model=args.base_model,
        lora_path=args.lora_path,
        track=args.track,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        use_vllm=use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )


if __name__ == '__main__':
    main()
