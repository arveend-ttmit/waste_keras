import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import threading
from collections import deque, Counter

import cv2
import numpy as np
import tf_keras
from flask import Flask, Response, render_template, jsonify
from PIL import Image

app = Flask(__name__)

BIN_MAP = {
    "Paper":         ("BIN 1", "Paper & cardboard", (17, 109, 59)),
    "Plastic_Metal": ("BIN 2", "Plastic & metal",   (11, 79, 133)),
    "Glass":         ("BIN 3", "Glass",             (165, 95, 24)),
    "Nothing":       (None,    "Nothing detected",  (90, 94, 95)),
}
CONFIDENCE_THRESHOLD = 0.70
ROI_FRACTION = 0.55      # ROI square = 55% of the shorter frame side
PREDICT_EVERY = 5        # run the model every Nth frame
SMOOTH_WINDOW = 7        # majority vote over the last N predictions

model = tf_keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

latest = {"ok": False, "bin": None, "label": "Starting...", "confidence": 0.0}
lock = threading.Lock()
history = deque(maxlen=SMOOTH_WINDOW)


def roi_box(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def preprocess_bgr(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def classify(crop):
    scores = model.predict(preprocess_bgr(crop), verbose=0)[0]
    idx = int(np.argmax(scores))
    return labels[idx], float(scores[idx])


def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)          # mirror, feels natural
        x1, y1, x2, y2 = roi_box(frame)

        if frame_no % PREDICT_EVERY == 0:
            label, conf = classify(frame[y1:y2, x1:x2])
            history.append(label if conf >= CONFIDENCE_THRESHOLD else "Nothing")
            voted = Counter(history).most_common(1)[0][0]
            bin_name, pretty, colour = BIN_MAP.get(voted, (None, voted, (90, 94, 95)))
            with lock:
                latest.update({
                    "ok": bin_name is not None,
                    "bin": bin_name,
                    "label": pretty,
                    "confidence": conf,
                })
        frame_no += 1

        with lock:
            colour = BIN_MAP.get(
                next((k for k, v in BIN_MAP.items() if v[1] == latest["label"]), "Nothing")
            )[2]
            caption = latest["bin"] or latest["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(frame, caption, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/result")
def result():
    with lock:
        return jsonify(dict(latest))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)