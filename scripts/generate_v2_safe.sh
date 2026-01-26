#!/bin/bash
#
# Safe Data Generation Script for GPSBench v2
# This script generates new data to data_v2/ without touching data/
#

set -e  # Exit on error

echo "================================================================================"
echo "GPSBench v2 Data Generation (Safe Mode)"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Create data_v2/ directory (current data/ untouched)"
echo "  2. Generate 3,400 samples per task with global coverage"
echo "  3. Create test splits (~1,000 samples per task)"
echo "  4. Generate comparison reports"
echo ""
echo "Current data/ directory will NOT be modified."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Create directory structure
echo ""
echo "Step 1/5: Creating data_v2/ directory structure..."
mkdir -p data_v2/track_pure_gps
mkdir -p data_v2/track_applied
mkdir -p data_v2/raw
mkdir -p comparison_reports

# Copy GeoNames data
if [ ! -f data_v2/raw/cities15000.txt ]; then
    cp data/raw/cities15000.txt data_v2/raw/
    echo "✅ Copied cities15000.txt"
fi

# Copy terrain and boundary data (needed for expansion)
if [ -f data/track_applied/task13_terrain_classification.json ]; then
    cp data/track_applied/task13_terrain_classification.json data_v2/track_applied/
    echo "✅ Copied terrain classification data (task13)"
fi

if [ -f data/track_applied/task8_boundary_analysis.json ]; then
    cp data/track_applied/task8_boundary_analysis.json data_v2/track_applied/
    echo "✅ Copied boundary analysis data (task8)"
fi

# Step 2: Backup current scripts and create v2 versions
echo ""
echo "Step 2/5: Preparing generation scripts..."

# Backup originals
cp scripts/generate_all_samples.py scripts/generate_all_samples.py.backup
cp scripts/generate_remaining_tasks.py scripts/generate_remaining_tasks.py.backup

# Modify in place (will restore after)
sed -i.tmp 's|data_path = base_path / "data"|data_path = base_path / "data_v2"|g' scripts/generate_all_samples.py
sed -i.tmp 's|data_path = base_path / "data"|data_path = base_path / "data_v2"|g' scripts/generate_remaining_tasks.py

echo "✅ Scripts modified to output to data_v2/"

# Step 3: Generate data
echo ""
echo "Step 3/5: Generating data (this may take 15-20 minutes)..."
echo ""

python3 scripts/generate_all_samples.py
echo ""
python3 scripts/generate_remaining_tasks.py

# Restore original scripts
echo ""
echo "Restoring original scripts..."
mv scripts/generate_all_samples.py.backup scripts/generate_all_samples.py
mv scripts/generate_remaining_tasks.py.backup scripts/generate_remaining_tasks.py
rm -f scripts/generate_all_samples.py.tmp scripts/generate_remaining_tasks.py.tmp

# Step 4: Create splits
echo ""
echo "Step 4/5: Creating train/dev/test splits..."

# Backup and modify create_splits.py
cp scripts/create_splits.py scripts/create_splits.py.backup
sed -i.tmp "s|data_dir = Path('data')|data_dir = Path('data_v2')|g" scripts/create_splits.py

python3 scripts/create_splits.py --track both --seed 42

# Restore original
mv scripts/create_splits.py.backup scripts/create_splits.py
rm -f scripts/create_splits.py.tmp

# Step 5: Verify and report
echo ""
echo "Step 5/5: Verifying generation..."
echo ""

# Count files
echo "Files generated:"
pure_gps_count=$(find data_v2/track_pure_gps -name "*.json" -not -path "*/splits/*" | wc -l)
applied_count=$(find data_v2/track_applied -name "*.json" -not -path "*/splits/*" | wc -l)
echo "  Pure GPS tasks: $pure_gps_count"
echo "  Applied tasks: $applied_count"

# Count test samples
echo ""
echo "Test set sizes (target: ~1000 per task):"
for f in data_v2/track_*/splits/*_test.json; do
    if [ -f "$f" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))")
        filename=$(basename "$f")
        echo "  $filename: $count samples"
    fi
done

# Total count
total_v2=$(find data_v2/track_*/splits/*_test.json -type f -exec python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" {} \; | awk '{s+=$1} END {print s}')
total_v1=$(find data/track_*/splits/*_test.json -type f -exec python3 -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" {} \; 2>/dev/null | awk '{s+=$1} END {print s}')

echo ""
echo "================================================================================"
echo "GENERATION COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Original (data/):     $total_v1 test samples"
echo "  New (data_v2/):       $total_v2 test samples"
echo "  Increase:             $((total_v2 - total_v1)) samples (+$(python3 -c "print(f'{$total_v2 / $total_v1:.1f}')") x)"
echo ""
echo "Next steps:"
echo "  1. Review data in data_v2/"
echo "  2. Run: python scripts/verify_global_coverage.py"
echo "  3. Compare: python scripts/compare_data_versions.py"
echo "  4. If satisfied, replace data/ with data_v2/"
echo ""
echo "Original data/ directory: UNTOUCHED ✅"
echo "================================================================================"
