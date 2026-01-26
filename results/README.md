# Evaluation Results Directory

This directory stores evaluation results in JSON format.

## File Naming Convention

Results are automatically saved with this naming pattern:
```
eval_{model_name}_{timestamp}.json
```

Examples:
- `eval_gpt-3.5-turbo_20251020_143052.json`
- `eval_gpt-4_20251020_144521.json`
- `eval_anthropic_claude-3-sonnet_20251020_150033.json`

## Result File Structure

Each result file contains:
```json
{
  "model": "gpt-4",
  "provider": "openai",
  "timestamp": "2025-10-20T14:35:52",
  "tasks": {
    "format_recognition": {
      "metrics": {
        "accuracy": 95.5,
        "correct_count": 764,
        "total_count": 800,
        ...
      },
      "test_results": [...]
    },
    ...
  },
  "overall_metrics": {
    "overall_accuracy": 87.3,
    "total_correct": 2033,
    "total_cases": 2329,
    ...
  }
}
```

## Viewing Results

### List All Results
```bash
python evaluation/list_results.py
```

Shows a table of all saved results with:
- Model name
- Provider
- Timestamp
- Overall accuracy
- Number of test cases

### Analyze Latest Result
```bash
python evaluation/analyze_results.py
```

Shows detailed breakdown of the most recent evaluation.

### Compare All Results
```bash
python evaluation/analyze_results.py --compare
```

Generates comparison table across all models evaluated.

### Analyze Specific Result
```bash
python evaluation/analyze_results.py --file eval_gpt-4_20251020_144521.json
```

### Export to CSV
```bash
python evaluation/analyze_results.py --compare --export-csv comparison.csv
```

Creates a CSV file suitable for spreadsheets or papers.

## Managing Results

### Keep Results Organized

Good practice:
1. Run evaluations with descriptive model names
2. Use `--output` flag for custom naming when needed
3. Regularly export comparisons to CSV
4. Archive old results periodically

Example with custom output name:
```bash
python evaluation/evaluate_layer_a.py \
  --model gpt-4 \
  --max-samples 100 \
  --output gpt4_baseline_100samples.json
```

### Backup Results

Results are valuable! Back them up:
```bash
# Create backup
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/*.json

# Or copy to cloud storage
cp results/*.json ~/Dropbox/GPSBench_Results/
```

### Clean Old Results

To remove old results but keep recent ones:
```bash
# Keep only last 10 results
ls -t results/eval_*.json | tail -n +11 | xargs rm
```

## Result Metrics Explained

### Overall Metrics
- **overall_accuracy**: Percentage correct across all tasks
- **total_correct**: Number of correct responses
- **total_cases**: Total test cases evaluated

### Task Metrics
- **accuracy**: Task-specific accuracy percentage
- **correct_count**: Correct responses for this task
- **error_count**: Number of API errors encountered
- **total_tokens**: Tokens used for this task
- **avg_latency_ms**: Average response time in milliseconds

### Test Results
Each test case includes:
- **input**: The coordinate string tested
- **response**: Model's response
- **correct**: Boolean flag
- **evaluation**: Detailed evaluation breakdown
- **tokens**: Tokens used for this test
- **latency_ms**: Response time

## Git Tracking

- ✅ This directory is tracked by git (via `.gitkeep`)
- ❌ Result files (`*.json`) are **not** tracked (see `.gitignore`)
- ✅ `README.md` is tracked

This keeps your repository clean while preserving directory structure.

## Cost Tracking

Each result file includes token usage:
```python
import json

with open('results/eval_gpt-4_20251020_144521.json') as f:
    data = json.load(f)

total_tokens = sum(
    task['metrics']['total_tokens']
    for task in data['tasks'].values()
    if 'metrics' in task
)

print(f"Total tokens: {total_tokens:,}")
```

## Common Analysis Tasks

### Find Best Model
```bash
python evaluation/analyze_results.py --compare
# Look at "Overall Best Model" section
```

### Find Task-Specific Performance
```bash
python evaluation/analyze_results.py --compare
# Look at "Best Models by Task" section
```

### Export for Paper
```bash
python evaluation/analyze_results.py \
  --compare \
  --export-csv paper_results.csv

# Then open in Excel/Google Sheets
```

### Track Improvement Over Time
```bash
# Run same model with same parameters over time
python evaluation/evaluate_layer_a.py --model gpt-4 --max-samples 100

# Compare results to see if model improved
python evaluation/analyze_results.py --compare
```

## Troubleshooting

### "No result files found"
- Run an evaluation first: `python evaluation/evaluate_layer_a.py --max-samples 10`
- Check you're in the right directory

### "Permission denied"
- Check file permissions: `ls -l results/`
- Make sure directory is writable

### "JSON decode error"
- File may be corrupted or incomplete
- Delete and re-run evaluation

## Tips

1. **Start small**: Use `--max-samples 10` for quick tests
2. **Compare wisely**: Don't compare results with different sample sizes
3. **Document context**: Note any special conditions in filename or separate notes
4. **Regular backups**: Results represent time and money invested
5. **Archive old results**: Keep your results directory manageable

---

Happy benchmarking! 📊
