# Evaluation

This directory contains the evaluation pipeline for NutriLens runs. Metrics are computed from stored run artifacts in `static/uploads/runs/<run_id>/result.json` and ground-truth annotations listed in `ground_truth.json`.

## Ground-truth schema

`ground_truth.json` must map run IDs to reference values:

```json
{
  "runs": {
    "<run_id>": {
      "volume_cm3": 320.5,
      "weight_g": 260.0,
      "calories": 410,
      "gt_mask_path": "path/to/gt_mask.png",
      "pred_mask_path": "path/to/pred_mask.png"
    }
  }
}
```

## Running evaluation

```bash
python -m evaluation.run_benchmark
```

Outputs are written to `evaluation/outputs/` and include CSV summaries and plots derived from computed results.
