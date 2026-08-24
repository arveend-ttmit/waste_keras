import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import math
import threading
from collections import deque, Counter

import cv2
import numpy as np
import tf_keras
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
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
MARGIN_MIN = 0.25

ROI_FRACTION = 0.55                # fallback static box, shown only when no hand is tracked
PREDICT_EVERY = 5
SMOOTH_WINDOW = 7

BOX_SCALE = 2.2
SMOOTH_ALPHA = 0.35
MIN_BOX_SIDE = 120

DETECT_EVERY = 2
DETECT_SCALE = 0.5

STABLE_NEEDED = 4                  # consecutive agreeing votes before locking/freezing
FREEZE_SECONDS = 1.2               # how long the "CONFIRMED" frame holds on screen
RESET_AFTER_S = 1.0                # hand must be absent this long before unlocking

GRASP_RATIO = 1.3                  # lower = stricter "must be gripping something"

ZONE_FRACTION = 0.6                # only hands centred in the middle 60% of frame count
MAX_HANDS = 3                      # detect up to this many hands per frame

AVG_WEIGHT_G = {
    "Paper":         20,
    "Plastic_Metal": 30,
    "Glass":        250,
}

# ---------- Model ----------
model = tf_keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", encoding="utf-8") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]

# ---------- Hand tracking (Tasks API) ----------
hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)
smoothed_box = None

# ---------- Shared state ----------
latest = {"ok": False, "bin": None, "label": "Starting...", "confidence": 0.0}
session = {"counts": {}, "total_items": 0, "est_weight_g": 0.0}
lock = threading.Lock()
history = deque(maxlen=SMOOTH_WINDOW)

locked_label = None      # None = searching, else = frozen/confirmed result
nothing_since = None

stream_lock = threading.Lock()   # prevents two simultaneous /video_feed connections


# ---------- Helpers ----------
def roi_box(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def in_zone(cx, cy, frame_w, frame_h):
    zx1, zy1 = frame_w * (1 - ZONE_FRACTION) / 2, frame_h * (1 - ZONE_FRACTION) / 2
    zx2, zy2 = frame_w - zx1, frame_h - zy1
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def zone_rect(frame_w, frame_h):
    zx1, zy1 = int(frame_w * (1 - ZONE_FRACTION) / 2), int(frame_h * (1 - ZONE_FRACTION) / 2)
    return zx1, zy1, frame_w - zx1, frame_h - zy1


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


def hand_roi_and_grasp(frame, timestamp_ms, prev_center):
    """
    Detect up to MAX_HANDS hands, keep only those inside the sorting zone,
    then pick ONE candidate to follow:
      1. whichever is closest to the hand we were already tracking (identity persistence)
      2. else prefer a hand that's actively grasping
      3. else the largest (closest-to-camera) hand
    Returns ((x1,y1,x2,y2) or None, is_grasping).
    """
    global smoothed_box
    h, w = frame.shape[:2]

    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if not res.hand_landmarks:
        smoothed_box = None
        return None, False

    candidates = []
    for landmarks in res.hand_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        cx, cy = pts.mean(axis=0)
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))

        if not in_zone(cx, cy, w, h):
            continue

        wrist = landmarks[0]
        tips, mcps = [4, 8, 12, 16, 20], [2, 5, 9, 13, 17]
        tip_dist = np.mean([math.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y) for i in tips])
        palm_dist = np.mean([math.hypot(landmarks[i].x - wrist.x, landmarks[i].y - wrist.y) for i in mcps])
        grasping = tip_dist < palm_dist * GRASP_RATIO

        candidates.append({"cx": cx, "cy": cy, "span": span, "grasping": grasping})

    if not candidates:
        smoothed_box = None
        return None, False

    if prev_center is not None:
        chosen = min(candidates, key=lambda c: math.hypot(c["cx"] - prev_center[0], c["cy"] - prev_center[1]))
    else:
        grasping_only = [c for c in candidates if c["grasping"]]
        pool = grasping_only if grasping_only else candidates
        chosen = max(pool, key=lambda c: c["span"])

    cx, cy = chosen["cx"], chosen["cy"]
    side = max(chosen["span"] * BOX_SCALE, MIN_BOX_SIDE)

    box = np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])
    smoothed_box = box if smoothed_box is None else (
        SMOOTH_ALPHA * box + (1 - SMOOTH_ALPHA) * smoothed_box
    )
    x1, y1, x2, y2 = smoothed_box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None, False

    return (x1, y1, x2, y2), chosen["grasping"]


# ---------- Main loop ----------
def generate_frames():
    global locked_label, nothing_since

    if not stream_lock.acquire(blocking=False):
        print("Stream already active — refusing second connection")
        return

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("camera opened:", cap.isOpened())

        for _ in range(10):
            cap.read()

        frame_no = 0
        fallback_box = None
        last_conf = 0.0
        last_box = None
        last_grasping = False
        last_center = None
        stable_count = 0
        freeze_until = 0.0
        freeze_img = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()

            if fallback_box is None:
                fallback_box = roi_box(frame)

            if frame_no % DETECT_EVERY == 0:
                last_box, last_grasping = hand_roi_and_grasp(frame, int(now * 1000), last_center)
                last_center = None
                if last_box is not None:
                    last_center = ((last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2)

            box = last_box
            tracking = box is not None
            x1, y1, x2, y2 = box if tracking else fallback_box

            if not tracking:
                history.append(NOTHING_LABEL)
                if nothing_since is None:
                    nothing_since = now
                elif now - nothing_since >= RESET_AFTER_S:
                    locked_label = None
                    stable_count = 0
                    nothing_since = None
            else:
                nothing_since = None

                if locked_label is not None:
                    pass  # frozen — skip detection/classify entirely

                elif not last_grasping:
                    history.append(NOTHING_LABEL)   # hand visible but not holding anything

                elif frame_no % PREDICT_EVERY == 0:
                    label, conf, margin = classify(frame[y1:y2, x1:x2])
                    last_conf = conf
                    ok_conf = conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_MIN
                    history.append(label if ok_conf else NOTHING_LABEL)

            if locked_label is None:
                voted = Counter(history).most_common(1)[0][0]
                if tracking and last_grasping and voted != NOTHING_LABEL and voted == history[-1]:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= STABLE_NEEDED:
                    locked_label = voted
                    freeze_until = now + FREEZE_SECONDS
                    freeze_img = frame.copy()

                    bin_name, pretty, _ = BIN_MAP.get(voted, (None, voted, None))
                    if bin_name is not None:
                        with lock:
                            session["counts"][bin_name] = session["counts"].get(bin_name, 0) + 1
                            session["total_items"] += 1
                            session["est_weight_g"] += AVG_WEIGHT_G.get(voted, 0)
            else:
                voted = locked_label

            bin_name, pretty, colour = BIN_MAP.get(voted, (None, voted, (90, 94, 95)))

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

            # ---- draw ----
            if now < freeze_until and freeze_img is not None:
                display = freeze_img.copy()
                cv2.rectangle(display, (x1, y1), (x2, y2), draw_colour, 4)
                cv2.putText(display, f"{caption}  \u2713 CONFIRMED",
                            (x1, max(20, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_colour, 2, cv2.LINE_AA)
            else:
                display = frame
                if not tracking:
                    label_text = "show me your hand"
                elif locked_label is None and not last_grasping:
                    label_text = "hold an item to scan"
                else:
                    label_text = caption
                thickness = 3 if tracking else 1
                cv2.rectangle(display, (x1, y1), (x2, y2), draw_colour, thickness)
                cv2.putText(display, label_text,
                            (x1, max(20, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_colour, 2, cv2.LINE_AA)

            zx1, zy1, zx2, zy2 = zone_rect(frame.shape[1], frame.shape[0])
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (70, 70, 75), 1)

            _, buf = cv2.imencode(".jpg", display)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

        cap.release()

    finally:
        stream_lock.release()


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