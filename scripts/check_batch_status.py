#!/usr/bin/env python3
"""Check the status of OpenAI batch jobs."""

import os
import sys
import time
from datetime import datetime
from openai import OpenAI

def format_timestamp(ts):
    """Format Unix timestamp to human-readable string."""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def format_duration(start_ts, end_ts=None):
    """Format duration between timestamps."""
    if start_ts is None:
        return "N/A"
    end = end_ts if end_ts else time.time()
    duration = int(end - start_ts)
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    seconds = duration % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def main():
    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Get batch ID from command line or list recent batches
    if len(sys.argv) > 1:
        batch_id = sys.argv[1]
        batches = [client.batches.retrieve(batch_id)]
        print(f"Checking batch: {batch_id}\n")
    else:
        print("Fetching recent batches...\n")
        batches = list(client.batches.list(limit=20).data)

    # Display batch information
    for i, batch in enumerate(batches):
        print(f"{'='*80}")
        print(f"Batch {i+1}: {batch.id}")
        print(f"{'='*80}")
        print(f"Status: {batch.status}")
        print(f"Endpoint: {batch.endpoint}")
        print(f"Created: {format_timestamp(batch.created_at)}")

        # Show duration based on status
        if batch.status == 'completed' and batch.completed_at:
            duration = format_duration(batch.created_at, batch.completed_at)
            print(f"Completed: {format_timestamp(batch.completed_at)} (took {duration})")
        elif batch.status in ['in_progress', 'validating']:
            duration = format_duration(batch.created_at)
            print(f"Running for: {duration}")
            if duration.endswith('m') or 'h' in duration:
                elapsed_mins = int(duration.split('m')[0].split()[-1]) if 'h' not in duration else 999
                if elapsed_mins > 10 and hasattr(batch, 'request_counts'):
                    rc = batch.request_counts
                    if rc.completed == 0 and rc.total > 0:
                        print(f"⚠️  WARNING: Batch may be stuck (no progress in {elapsed_mins} minutes)")

        if hasattr(batch, 'metadata') and batch.metadata:
            print(f"Metadata: {batch.metadata}")

        if hasattr(batch, 'request_counts') and batch.request_counts:
            rc = batch.request_counts
            print(f"Requests: {rc.completed}/{rc.total} completed, {rc.failed} failed")
            if rc.total > 0:
                progress = (rc.completed / rc.total) * 100
                print(f"Progress: {progress:.1f}%")

        print(f"Output file: {batch.output_file_id or 'None'}")
        print(f"Error file: {batch.error_file_id or 'None'}")

        if batch.errors:
            print(f"Batch errors: {batch.errors}")

        # If there's an error file, download and show it
        if batch.error_file_id:
            print("\n--- Error File Contents ---")
            try:
                error_content = client.files.content(batch.error_file_id).content
                error_text = error_content.decode('utf-8')
                print(error_text[:1000])  # First 1000 chars
                if len(error_text) > 1000:
                    print(f"\n... (truncated, {len(error_text)} total chars)")
            except Exception as e:
                print(f"Error reading error file: {e}")

        # Suggest actions
        if batch.status in ['in_progress', 'validating']:
            print(f"\n💡 To cancel: python scripts/cancel_batch.py {batch.id}")

        if i < len(batches) - 1:
            print()  # Empty line between batches

if __name__ == "__main__":
    main()
