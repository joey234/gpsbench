#!/bin/bash
# Add error metrics to benchmark results
#
# Usage:
#   ./scripts/add_error_metrics.sh <results_folder>
#
# Example:
#   ./scripts/add_error_metrics.sh results/gpsbench_gpt-5-mini_20251205_142538

if [ $# -eq 0 ]; then
    echo "Error: No results folder specified"
    echo ""
    echo "Usage: ./scripts/add_error_metrics.sh <results_folder>"
    echo ""
    echo "Example:"
    echo "  ./scripts/add_error_metrics.sh results/gpsbench_gpt-5-mini_20251205_142538"
    exit 1
fi

RESULTS_FOLDER=$1

echo "================================================================================"
echo "Adding Error Metrics to Results"
echo "================================================================================"
echo "Results folder: $RESULTS_FOLDER"
echo ""

# Step 1: Compute error metrics
echo "Step 1: Computing error metrics..."
python3 scripts/compute_error_metrics.py "$RESULTS_FOLDER"

if [ $? -ne 0 ]; then
    echo "Error: Failed to compute error metrics"
    exit 1
fi

echo ""

# Step 2: Update summary and README
echo "Step 2: Updating summary and README..."
python3 scripts/update_summary_with_errors.py "$RESULTS_FOLDER"

if [ $? -ne 0 ]; then
    echo "Error: Failed to update summary and README"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ Done! Error metrics have been added to:"
echo "   - $RESULTS_FOLDER/error_metrics.json"
echo "   - $RESULTS_FOLDER/summary.json (updated)"
echo "   - $RESULTS_FOLDER/README.md (updated)"
echo "================================================================================"
