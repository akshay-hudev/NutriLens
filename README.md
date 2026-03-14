# NutriLens 🍽️

Real-time visual food calorie estimation using deep learning.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> On Apple Silicon (M1/M2), replace `tensorflow` with `tensorflow-macos tensorflow-metal`

### 2. Add the trained model
Train `model_trained_101class.hdf5` in Google Colab using `Food_Image_Recognition.ipynb`,
then place it in this folder. Without it, the app runs in **demo mode** (random predictions).

### 3. (Optional) Set Gemini API key
Get a free key at https://aistudio.google.com and set it:
```bash
export GOOGLE_API_KEY="your_key_here"
```

### 4. Run
```bash
python app.py
```
Open http://127.0.0.1:5000

```bash
# Debug mode
python app.py --debug

# Custom host/port
python app.py 0.0.0.0 8080
```

## Project Structure

```
NutriLens/
├── app.py                        # Flask app (fixed & improved)
├── nutrition101.csv              # USDA nutrition data for 101 foods
├── requirements.txt
├── model_trained_101class.hdf5   # ← Add this from Colab training
├── get_nutrition_data.py         # Script to regenerate nutrition CSV
├── Food_Image_Recognition.ipynb  # Colab training notebook
├── static/
│   └── uploads/                  # Auto-created on first run
└── templates/
    ├── index.html                # Homepage with drag-and-drop upload
    ├── recognize.html            # Upload confirmation
    └── results.html              # Nutritional results report
```

## Model Info

- Architecture: InceptionV3 (transfer learning, ImageNet weights)
- Dataset: Food-101 (101,000 images, 101 classes)
- Training: 30 epochs, ~10–11 hrs on Google Colab T4 GPU
- Val accuracy: ~77%

## Tech Stack

- **ML**: TensorFlow / Keras, InceptionV3
- **Backend**: Flask (Python)
- **Nutrition data**: USDA FoodData Central API
- **AI descriptions**: Google Gemini Vision (optional)
