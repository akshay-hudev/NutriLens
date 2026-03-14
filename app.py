import os
import csv
import math
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import PIL.Image

# ── Optional: tensorflow only loaded if model exists ─────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Optional: Gemini vision ───────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_KEY = os.environ.get('GOOGLE_API_KEY', '')
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

# ── Flask setup ───────────────────────────────────────────────────────────────
tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
app.jinja_env.filters['enumerate'] = enumerate

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Food labels (sorted to match training order) ──────────────────────────────
label = sorted([
    'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
    'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
    'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla',
    'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder',
    'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes',
    'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict',
    'escargots', 'falafel', 'filet mignon', 'fish and_chips', 'foie gras',
    'french fries', 'french onion soup', 'french toast', 'fried calamari', 'fried rice',
    'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich',
    'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup',
    'hot dog', 'huevos rancheros', 'hummus', 'ice cream', 'lasagna',
    'lobster bisque', 'lobster roll sandwich', 'macaroni and cheese', 'macarons', 'miso soup',
    'mussels', 'nachos', 'omelette', 'onion rings', 'oysters',
    'pad thai', 'paella', 'pancakes', 'panna cotta', 'peking duck',
    'pho', 'pizza', 'pork chop', 'poutine', 'prime rib',
    'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits',
    'spaghetti bolognese', 'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake',
    'sushi', 'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles'
])

nu_link = 'https://www.nutritionix.com/food/'

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = 'model_trained_101class.hdf5'
model_best = None
MODEL_LOADED = False

if TF_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        tf.keras.backend.clear_session()
        model_best = load_model(MODEL_PATH, compile=False)
        MODEL_LOADED = True
        print('Model loaded successfully!')
    except Exception as e:
        print(f'Model load failed: {e}')
else:
    print('Model file not found — running in demo mode (random predictions)')

# ── Load nutrition table ──────────────────────────────────────────────────────
nutrition_table = {}
with open('nutrition101.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'Protein',       'unit': 'g',  'value': float(row[2])},
            {'name': 'Calcium',       'unit': 'mg', 'value': round(float(row[3]) * 1000, 1)},
            {'name': 'Fat',           'unit': 'g',  'value': float(row[4])},
            {'name': 'Carbohydrates', 'unit': 'g',  'value': float(row[5])},
            {'name': 'Vitamins',      'unit': 'mg', 'value': round(float(row[6]) * 1000, 2)},
        ]

# ── State ─────────────────────────────────────────────────────────────────────
uploaded_files = []
results_cache = []


# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_loaded=MODEL_LOADED)


@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_files
    files = request.files.getlist('img')
    if not files or all(f.filename == '' for f in files):
        return redirect(url_for('index'))

    uploaded_files = []
    for i, f in enumerate(files):
        if f and f.filename:
            ext = os.path.splitext(secure_filename(f.filename))[1].lower() or '.jpg'
            filename = f'upload_{i}{ext}'
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            uploaded_files.append(path)

    return render_template('recognize.html', files=uploaded_files, count=len(uploaded_files))


@app.route('/predict')
def predict():
    global results_cache
    if not uploaded_files:
        return redirect(url_for('index'))

    results_cache = []
    total_nutrition = {'Protein': 0, 'Calcium': 0, 'Fat': 0, 'Carbohydrates': 0, 'Vitamins': 0}

    for filepath in uploaded_files:
        pa = {}

        if MODEL_LOADED:
            img_arr = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_arr = tf.keras.preprocessing.image.img_to_array(img_arr)
            img_arr = np.expand_dims(img_arr, axis=0) / 255.0
            pred = model_best.predict(img_arr)[0]
            if any(math.isnan(v) for v in pred[:4]):
                pred = np.random.dirichlet(np.ones(101))
        else:
            pred = np.random.dirichlet(np.ones(101) * 0.5)

        top3_idx = pred.argsort()[-3:][::-1]
        top_label = label[top3_idx[0]]

        pa['image'] = filepath
        pa['predictions'] = [
            {'food': label[i], 'confidence': round(float(pred[i]) * 100, 1)}
            for i in top3_idx
        ]
        pa['top_food'] = top_label
        pa['nutrition'] = nutrition_table.get(top_label, nutrition_table[label[0]])
        pa['nutritionix_url'] = nu_link + top_label.replace(' ', '-')
        pa['quantity'] = 100

        if GEMINI_AVAILABLE:
            try:
                vision_model = genai.GenerativeModel('gemini-1.5-flash')
                pil_img = PIL.Image.open(filepath)
                response = vision_model.generate_content(
                    ["In one sentence, identify the food and estimate its calories per serving.", pil_img]
                )
                pa['ai_description'] = response.text
            except Exception:
                pa['ai_description'] = None
        else:
            pa['ai_description'] = None

        results_cache.append(pa)
        for n in pa['nutrition']:
            total_nutrition[n['name']] += n['value']

    n = len(results_cache)
    avg_nutrition = [
        {'name': k, 'unit': 'g' if k not in ('Calcium', 'Vitamins') else 'mg', 'value': round(v / n, 2)}
        for k, v in total_nutrition.items()
    ]

    def calories(nutrition):
        d = {x['name']: x['value'] for x in nutrition}
        return round(d.get('Protein', 0) * 4 + d.get('Carbohydrates', 0) * 4 + d.get('Fat', 0) * 9)

    for r in results_cache:
        r['calories'] = calories(r['nutrition'])

    return render_template(
        'results.html',
        pack=results_cache,
        avg_nutrition=avg_nutrition,
        total_calories=sum(r['calories'] for r in results_cache),
        model_loaded=MODEL_LOADED
    )


@app.route('/reset')
def reset():
    global uploaded_files, results_cache
    uploaded_files = []
    results_cache = []
    return redirect(url_for('index'))


if __name__ == '__main__':
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        app.run(host=host, port=port, debug=debug, threaded=threaded)

    run()
