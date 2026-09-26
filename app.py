from pathlib import Path
from datetime import datetime
from uuid import uuid4

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"

ALLOWED_EXTENSIONS = {"jpeg", "jpg", "png"}
IMAGE_SIZE = (224, 224)
MAX_HISTORY = 5
prediction_history = []

app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
model = MobileNetV2(weights="imagenet")


def allowed_file(filename):
    """Return whether the filename has an allowed image extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_image(image_path):
    """Preprocess an image and return its top ImageNet prediction."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32)

    image_batch = np.expand_dims(image_array, axis=0)
    image_batch = preprocess_input(image_batch)
    predictions = model.predict(image_batch, verbose=0)
    _, label, confidence = decode_predictions(predictions, top=1)[0][0]

    return {
        "label": label.replace("_", " "),
        "confidence": confidence * 100,
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return render_template(
        "index.html",
        error="The image is too large. Please upload an image smaller than 5 MB.",
        history=prediction_history,
    ), 413


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please choose an image to classify."
        else:
            safe_original_name = secure_filename(uploaded_file.filename)
            if not safe_original_name or not allowed_file(safe_original_name):
                    error = "Please upload a supported image file (PNG, JPG, or JPEG)."
            else:
                extension = Path(safe_original_name).suffix.lower()
                temporary_path = app.config["UPLOAD_FOLDER"] / f"{uuid4().hex}{extension}"
                uploaded_file.save(temporary_path)

                try:
                    prediction = classify_image(temporary_path)
                    prediction_history.insert(
                        0,
                        {
                            "filename": safe_original_name,
                            "label": prediction["label"],
                            "confidence": prediction["confidence"],
                            "time": datetime.now().strftime("%b %d, %Y · %H:%M"),
                        },
                    )
                    del prediction_history[MAX_HISTORY:]
                except (UnidentifiedImageError, OSError):
                    error = "That file could not be read as a valid image."
                except Exception:
                    app.logger.exception("Image classification failed")
                    error = "The image could not be classified. Please try another image."
                finally:
                    temporary_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        history=prediction_history,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)