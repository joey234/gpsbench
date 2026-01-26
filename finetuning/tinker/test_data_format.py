#!/usr/bin/env python3
"""
Quick test to verify training data format includes actual coordinates/points
"""

import json
import sys
sys.path.insert(0, '.')

from gpsbench_datasets import GPSBenchSplitDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Create minimal config
config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Instruct-2507",
    renderer_name="qwen3_instruct",
    max_length=16384,
    batch_size=64,
)

# Create dataset builder
dataset = GPSBenchSplitDataset(
    common_config=config,
    gpsbench_root="../../",
    track="pure_gps",
    tasks=["task1_format_conversion", "task4_distance_calculation"],
    max_samples=3,
    max_eval_samples=2,
)

# Load conversations
print("=" * 80)
print("TESTING DATA FORMAT - Loading first 2 examples")
print("=" * 80)

conversations = dataset.load_conversations("train")

for i, conv in enumerate(conversations[:2], 1):
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i} - Task: {conv.get('task', 'unknown')}")
    print(f"{'='*80}")

    messages = conv.get('messages', [])
    for msg in messages:
        role = msg['role']
        content = msg['content']
        print(f"\n[{role.upper()}]")
        print(content)
        print()

print("\n" + "=" * 80)
print("VERIFICATION CHECKLIST:")
print("=" * 80)
print("✓ Check that format_conversion includes source_string (the actual coordinate)")
print("✓ Check that distance_calculation includes lat/lon for both points")
print("✓ Check that place_association includes coordinate_string")
print("=" * 80)
