"""Flask web application for handwritten digit recognition."""
from pathlib import Path
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run `python src/main.py` to train and save the model."
    )

model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="Image is too large. Maximum size is 10 MB."), 413


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return render_template("index.html", error="Please select an image.")

    if not allowed_file(uploaded.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Upload PNG, JPG, JPEG, BMP, or WEBP."
        )

    filename = f"{uuid.uuid4().hex}.png"
    destination = UPLOAD_DIR / filename

    try:
        with Image.open(uploaded.stream) as original:
            image = original.convert("L").resize((28, 28))
            image_array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template("index.html", error="The selected file is not a valid image.")

    # MNIST uses light digits on a dark background. Invert typical dark ink
    # on white paper to match that format.
    if image_array.mean() > 0.5:
        image_array = 1.0 - image_array

    image_array = image_array.reshape(1, 28, 28, 1)
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = round(float(np.max(probabilities) * 100), 2)

    # Store a preview copy in Flask's static directory.
    image.save(destination, format="PNG")

    return render_template(
        "index.html",
        prediction=predicted_digit,
        confidence=confidence,
        image_url=f"uploads/{filename}",
    )


if __name__ == "__main__":
    app.run(debug=True)
