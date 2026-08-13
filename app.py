import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import threading
from collections import deque, Counter

import cv2
import numpy as np
import tf_keras
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify
from PIL import Image

app = Flask(__name__)

# ---------- Config ----------
BIN_MAP = {
    "Paper":         ("BIN 1", "Paper & cardboard", (17, 109, 59)),
    "Plastic_Metal": ("BIN 2", "Plastic & metal",   (11, 79, 133)),
    "Glass":         ("BIN 3", "Glass",             (165, 95, 24)),
    "Nothing":       (None,    "Nothing detected",  (90, 94, 95)),
}
NOTHING_LABEL = "Nothing"          # must match labels.txt exactly
CONFIDENCE_THRESHOLD = 0.70
MARGIN_MIN = 0.25                  # top1 - top2 score gap required to trust a prediction

ROI_FRACTION = 0.55                # fallback static box, used only if no hand is found yet
PREDICT_EVERY = 5                  # run the model every Nth frame
SMOOTH_WINDOW = 7                  # majority vote over the last N predictions

BOX_SCALE = 2.2                    # hand-crop box size relative to hand span
SMOOTH_ALPHA = 0.35                # EMA smoothing for the tracked box
MIN_BOX_SIDE = 120

RESET_AFTER_S = 1.0                # how long "Nothing" must persist before session resets

AVG_WEIGHT_G = {
    "Paper":         20,
    "Plastic_Metal": 30,
    "Glass":        250,
}

# ---------- Model ----------
model = tf_keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

# ---------- Hand tracking ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
smoothed_box = None

# ---------- Shared state ----------
latest = {"ok": False, "bin": None, "label": "Starting...", "confidence": 0.0}
session = {"counts": {}, "total_items": 0, "est_weight_g": 0.0}
lock = threading.Lock()
history = deque(maxlen=SMOOTH_WINDOW)

item_present = False
nothing_since = None


# ---------- Helpers ----------
def roi_box(frame):
    """Static centred square — used as a fallback guide box only."""
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
    order = np.argsort(scores)[::-1]
    top, second = float(scores[order[0]]), float(scores[order[1]])
    return labels[int(order[0])], top, top - second


def hand_roi(frame):
    """Return (x1, y1, x2, y2) centred on the detected hand, or None if no hand."""
    global smoothed_box
    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not res.multi_hand_landmarks:
        smoothed_box = None
        return None

    pts = np.array([[lm.x * w, lm.y * h] for lm in res.multi_hand_landmarks[0].landmark])
    cx, cy = pts.mean(axis=0)
    side = max(pts[:, 0].ptp(), pts[:, 1].ptp()) * BOX_SCALE
    side = max(side, MIN_BOX_SIDE)

    box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
    smoothed_box = box if smoothed_box is None else (
        SMOOTH_ALPHA * box + (1 - SMOOTH_ALPHA) * smoothed_box
    )

    x1, y1, x2, y2 = smoothed_box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return x1, y1, x2, y2


# ---------- Main loop ----------
def generate_frames():
    global item_present, nothing_since

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("camera opened:", cap.isOpened())

    for _ in range(10):        # discard warm-up frames
        cap.read()

    frame_no = 0
    fallback_box = None
    last_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        if fallback_box is None:
            fallback_box = roi_box(frame)

        box = hand_roi(frame)
        tracking = box is not None
        x1, y1, x2, y2 = box if tracking else fallback_box

        if not tracking:
            history.append(NOTHING_LABEL)     # no hand at all → definitely nothing
        elif frame_no % PREDICT_EVERY == 0:
            label, conf, margin = classify(frame[y1:y2, x1:x2])
            last_conf = conf
            ok_conf = conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_MIN
            history.append(label if ok_conf else NOTHING_LABEL)

        voted = Counter(history).most_common(1)[0][0]
        bin_name, pretty, colour = BIN_MAP.get(voted, (None, voted, (90, 94, 95)))

        now = time.time()
        if bin_name is not None:
            nothing_since = None
            if not item_present:
                item_present = True
                with lock:
                    session["counts"][bin_name] = session["counts"].get(bin_name, 0) + 1
                    session["total_items"] += 1
                    session["est_weight_g"] += AVG_WEIGHT_G.get(voted, 0)
        else:
            if nothing_since is None:
                nothing_since = now
            elif now - nothing_since >= RESET_AFTER_S:
                item_present = False

        with lock:
            latest.update({
                "ok": bin_name is not None,
                "bin": bin_name,
                "label": pretty,
                "confidence": last_conf if tracking else 0.0,
            })
            caption = latest["bin"] or latest["label"]
            draw_colour = colour if tracking else (90, 94, 95)

        frame_no += 1

        thickness = 3 if tracking else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_colour, thickness)
        cv2.putText(frame, caption if tracking else "show me your hand",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_colour, 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()


# ---------- Routes ----------
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


@app.route("/session")
def get_session():
    with lock:
        return jsonify(dict(session))


@app.route("/session/reset", methods=["POST"])
def reset_session():
    with lock:
        session.update({"counts": {}, "total_items": 0, "est_weight_g": 0.0})
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)