import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import io
import numpy as np
import tf_keras
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

app = Flask(__name__)

BIN_MAP = {
    "Paper":         ("BIN 1", "Paper & cardboard"),
    "Plastic_Metal": ("BIN 2", "Plastic & metal"),
    "Glass":         ("BIN 3", "Glass"),
    "Nothing":       (None,    "Nothing detected"),
}
CONFIDENCE_THRESHOLD = 0.70

model = tf_keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    scores = model.predict(preprocess(img), verbose=0)[0]
    idx = int(np.argmax(scores))
    label, confidence = labels[idx], float(scores[idx])
    bin_name, pretty = BIN_MAP.get(label, (None, label))

    if bin_name is None or confidence < CONFIDENCE_THRESHOLD:
        return jsonify({"ok": False, "label": pretty, "confidence": confidence})
    return jsonify({"ok": True, "bin": bin_name, "label": pretty, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)