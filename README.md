# NutriLens

Research-style Flask prototype for monocular 3D food volume reconstruction and nutrition estimation.

The app accepts 1-5 sparse food views, segments the food region, estimates camera geometry, reconstructs a lightweight Gaussian-splat-style proxy, estimates volume in cm3, converts volume to portion weight with food density priors, and scales USDA nutrition values to the estimated portion.

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

Place `model_trained_101class.hdf5` in the project root, or set `NUTRILENS_MODEL_PATH`.

If the classifier is missing or TensorFlow is unavailable, NutriLens no longer returns random predictions. It uses a deterministic low-confidence visual fallback and clearly marks the result as uncertain.

`requirements.txt` installs TensorFlow only on Python versions supported by the pinned classifier runtime. On newer Python versions, the Flask app still runs and uses the fallback unless you install a compatible TensorFlow build separately.

Native COLMAP, Gaussian Splatting, and NeRF integrations are optional. When they are not configured, the app still runs a transparent silhouette-depth reconstruction proxy and widens the volume uncertainty interval.

## Optional Environment Variables

Copy `.env.example` to `.env` and adjust values.

Key settings:

- `GOOGLE_API_KEY`: optional Gemini food description.
- `NUTRILENS_MODEL_PATH`: Food-101 classifier path.
- `NUTRILENS_NUTRITION_CSV`: nutrition table path.
- `NUTRILENS_RECON_BACKEND`: `gsplat` by default, `nerf` hook available.
- `NUTRILENS_ENABLE_COLMAP`: run COLMAP preprocessing when installed.
- `NUTRILENS_BENCHMARK_DATA_ROOT`: MetaFood3D-style benchmark hook.

## Project Structure

```text
NutriLens/
  app.py
  config.py
  nutrition101.csv
  get_nutrition_data.py
  reconstruction/
    colmap.py
    gsplat.py
    nerf.py
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
```

## Research Direction

The code is structured so future experiments can compare sparse-view 3D volume estimates against 2D area baselines and MetaFood3D-style ground truth with images, masks, depth maps, meshes, and nutrition labels.
