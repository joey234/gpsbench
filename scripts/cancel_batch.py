#!/usr/bin/env python3
"""Cancel an OpenAI batch job."""

import os
import sys
from openai import OpenAI

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/cancel_batch.py <batch_id>")
        print("\nExample:")
        print("  python scripts/cancel_batch.py batch_692d86...")
        sys.exit(1)

    batch_id = sys.argv[1]

    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Get batch status first
    print(f"Fetching batch {batch_id}...")
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"\nCurrent status: {batch.status}")
        if hasattr(batch, 'request_counts'):
            rc = batch.request_counts
            print(f"Requests: {rc.completed}/{rc.total} completed, {rc.failed} failed")
    except Exception as e:
        print(f"Error fetching batch: {e}", file=sys.stderr)
        sys.exit(1)

    # Confirm cancellation
    if batch.status not in ['validating', 'in_progress']:
        print(f"\n⚠️  Batch is in '{batch.status}' state - cancellation may not be necessary")

    response = input(f"\nCancel batch {batch_id}? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancellation aborted.")
        sys.exit(0)

    # Cancel batch
    print(f"\nCancelling batch {batch_id}...")
    try:
        cancelled_batch = client.batches.cancel(batch_id)
        print(f"✅ Batch cancellation initiated")
        print(f"Status: {cancelled_batch.status}")
        print(f"\nNote: It may take up to 10 minutes for in-flight requests to complete.")
    except Exception as e:
        print(f"❌ Error cancelling batch: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
