# NutriLens

Research-oriented Flask prototype for sparse-view 3D food volume reconstruction and nutrition estimation.

NutriLens accepts 1-5 food views, segments the food region, estimates camera poses, reconstructs a lightweight Gaussian-splat proxy, estimates volume in cm3, converts volume to portion weight using food density priors, and scales USDA nutrition values to the estimated portion. The system exposes uncertainty and confidence labels at each stage to make results transparent for research evaluation.

## Abstract

NutriLens is a modular, research-style pipeline that combines 2D food recognition, segmentation, sparse-view geometry, and nutritional scaling. The goal is to approximate real-world portion size from a small set of images, then translate that geometry into nutrition estimates with explicit confidence and uncertainty bounds. The codebase is structured to support ablation studies and future integration of COLMAP, native Gaussian Splatting, and NeRF backends.

## System Pipeline

1. Upload 1-5 food images.
2. Classify the food using a Food-101 classifier, or a deterministic low-confidence fallback if TensorFlow/model weights are missing.
3. Segment the food region using GrabCut when OpenCV is available, otherwise use a color-center heuristic mask.
4. Estimate camera poses using COLMAP if enabled, otherwise use an ordered sparse-view pose prior.
5. Reconstruct a Gaussian-splat proxy point cloud and derive a geometry estimate (width, depth, height).
6. Estimate volume with an ellipsoid-cap approximation and compute uncertainty bounds.
7. Convert volume to portion weight using density priors and scale USDA nutrition values.
8. Optionally generate a Gemini description for user-facing context (not used for nutrition).

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

```bash
# Debug mode
python app.py --debug

# Custom host/port
python app.py 0.0.0.0 8080
```

## Model And Fallbacks

- Place model_trained_101class.hdf5 in the project root, or set NUTRILENS_MODEL_PATH.
- If the classifier is missing or TensorFlow is unavailable, NutriLens uses a deterministic low-confidence fallback and flags the result as uncertain.
- TensorFlow 2.13 is pinned only for Python versions that support it. On newer Python versions, the Flask app runs with the fallback unless you install a compatible TensorFlow build.
- COLMAP, native gsplat, and NeRF integrations are optional. When not configured, the system runs a silhouette-depth reconstruction proxy and widens the volume uncertainty interval.

## Outputs And Artifacts

Each run writes a self-contained folder under static/uploads/runs/<run_id>/:

- manifest.json: uploaded images and warnings
- images/: normalized JPEG inputs
- masks/: segmentation masks and overlays
- reconstruction/: proxy point cloud (food_proxy_gaussians.ply)
- reconstruction/: voxel occupancy grid (food_proxy_voxels.npz)
- result.json: full report (classification, segmentation, poses, reconstruction, volume, portion, and warnings)

This layout makes it easy to archive runs for qualitative or quantitative evaluation.

## Configuration

Copy .env.example to .env and adjust values as needed. Key settings:

- GOOGLE_API_KEY, NUTRILENS_ENABLE_GEMINI, NUTRILENS_GEMINI_MODEL
- NUTRILENS_MODEL_PATH, NUTRILENS_NUTRITION_CSV
- NUTRILENS_MAX_IMAGES, NUTRILENS_MAX_IMAGE_BYTES
- NUTRILENS_FRAME_DIAMETER_CM, NUTRILENS_PLATE_DIAMETER_CM
- NUTRILENS_DEFAULT_DENSITY
- NUTRILENS_ENABLE_PLATE_DETECTION
- NUTRILENS_REFERENCE_DIAMETER_CM, NUTRILENS_REFERENCE_DIAMETER_PX
- NUTRILENS_SCALE_JSON (optional run-level scale metadata file name)
- NUTRILENS_SEGMENTATION_BACKEND (auto, grabcut, heuristic, deeplabv3, u2net, sam)
- NUTRILENS_SEGMENTATION_DEVICE (cpu by default for torch backends)
- NUTRILENS_U2NET_ONNX, NUTRILENS_U2NET_INPUT_SIZE
- NUTRILENS_SAM_CHECKPOINT, NUTRILENS_SAM_MODEL_TYPE
- NUTRILENS_VOXEL_SIZE_CM, NUTRILENS_VOXEL_MAX_DIM, NUTRILENS_COLMAP_POINT_LIMIT
- NUTRILENS_RECON_BACKEND (gsplat or nerf)
- NUTRILENS_ENABLE_COLMAP, COLMAP_BIN
- NUTRILENS_ENABLE_GSPLAT, NUTRILENS_ENABLE_NERF
- NUTRILENS_RECON_TIMEOUT_SECONDS
- NUTRILENS_BENCHMARK_NAME, NUTRILENS_BENCHMARK_DATA_ROOT

## Optional Dependencies

- tensorflow: Food-101 classifier runtime (optional)
- opencv-python: GrabCut segmentation (optional)
- google-generativeai: Gemini description (optional)
- colmap: camera pose estimation (optional)
- torch: NeRF hooks (optional)

## Research Dataset Hook

Set NUTRILENS_BENCHMARK_DATA_ROOT to a MetaFood3D-style directory to enable dataset metadata in reports. Expected layout:

```
<root>/
  images/
  masks/
  depth/
  meshes/
  nutrition_labels.csv
```

## Project Structure

```
NutriLens/
  app.py
  config.py
  nutrition101.csv
  get_nutrition_data.py
  requirements.txt
  NutriLens_Training.ipynb
  paper/
  reconstruction/
    colmap.py
    gsplat.py
    nerf.py
    types.py
    volume.py
  services/
    classification_service.py
    description_service.py
    nutrition_service.py
    reconstruction_service.py
    research_dataset_service.py
    segmentation_service.py
  utils/
    image_io.py
    volume_math.py
  templates/
    index.html
    recognize.html
    results.html
  static/
    uploads/
```

## Design Notes And Limitations

- The reconstruction stage is a proxy meant for research iteration, not a production-grade 3D model.
- Volume estimation relies on a calibrated frame-diameter prior rather than explicit metric calibration.
- Segmentation defaults to heuristics without a dedicated food segmenter.
- Density priors are coarse and should be replaced with learned food-specific densities for high-accuracy studies.
- The optional Gemini description is user-facing only and does not influence the nutrition estimate.

## Data Sources

- Food-101 labels for classification categories.
- USDA FoodData Central for nutrition values (nutrition101.csv).

## Reproducing The Nutrition Table

If you need to regenerate nutrition101.csv, obtain a USDA API key and run:

```bash
python get_nutrition_data.py
```

## Evaluation Pipeline

NutriLens includes an evaluation toolkit that computes metrics from stored run artifacts.
Populate evaluation/ground_truth.json with reference values (volume, weight, calories, masks), then run:

```bash
python -m evaluation.run_benchmark
```

This produces evaluation/outputs/per_run.csv and evaluation/outputs/summary.csv. Paper figures are generated from these outputs; no benchmark numbers are hardcoded in the plotting scripts.

## Research Direction

The code is structured to compare sparse-view 3D volume estimates against 2D area baselines and benchmark datasets with images, masks, depth maps, meshes, and nutrition labels. The reconstruction and dataset services are organized so new backends and evaluation metrics can be added with minimal changes to the Flask interface.
