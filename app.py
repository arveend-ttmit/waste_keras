"""
Waste sorting demo - Flask + MediaPipe hand tracking.

Two classifier backends, switchable at runtime:

  cloud  - one call to gemma4:31b-cloud per item presented (open vocabulary).
  local  - the original Teachable Machine keras_model.h5 (offline fallback).

    set WASTE_BACKEND=local        force the offline path
    set WASTE_DEBUG=1              dump raw cloud responses (default on)

Diagnostics - start here when the cloud path fails:
    GET  /diag      4-stage self-test; the first failing stage is your answer
    GET  /lastraw   the last raw model message, verbatim
    GET  /health    quick reachability check
    POST /backend/local | /backend/cloud

Stages:
    1 reach   - can we talk to Ollama at all, is the model listed
    2 text    - does it reply with plain text (rules out thinking-mode)
    3 vision  - does the image actually arrive
    4 schema  - does structured output come back parseable
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import base64
import hashlib
import json
import math
import threading
import time
import traceback
from collections import deque, Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

app = Flask(__name__)

# ---------------------------------------------------------------- config ----
BACKEND = os.environ.get("WASTE_BACKEND", "cloud").lower()
DEBUG_CLOUD = os.environ.get("WASTE_DEBUG", "1") == "1"

CLOUD_MODEL = "gemma4:31b-cloud"
CLOUD_TIMEOUT_S = 40
CLOUD_MIN_CONFIDENCE = 0.35
CLOUD_NUM_PREDICT = 512          # too small => empty content
CACHE_RESPONSES = True
DEBUG_DIR = Path("debug")

BIN_MAP = {
    "BIN 1": ("Paper & cardboard", (17, 109, 59)),
    "BIN 2": ("Plastic & metal", (11, 79, 133)),
    "BIN 3": ("Glass", (165, 95, 24)),
}
GREY = (90, 94, 95)

LOCAL_BIN = {"Paper": "BIN 1", "Plastic_Metal": "BIN 2", "Glass": "BIN 3"}
NOTHING_LABEL = "Nothing"
AVG_WEIGHT_G = {"BIN 1": 20, "BIN 2": 30, "BIN 3": 250}

ROI_FRACTION = 0.55
BOX_SCALE = 2.2
SMOOTH_ALPHA = 0.35
MIN_BOX_SIDE = 120
DETECT_EVERY = 2
DETECT_SCALE = 0.5
ZONE_FRACTION = 0.95
MAX_HANDS = 3
GRASP_RATIO = 1.6
OBJECT_FILL_MIN = 0.35

STEADY_FRAMES = 8
STEADY_PX = 25
MIN_GAP_S = 2.5
RESET_AFTER_S = 1.0

CONFIDENCE_THRESHOLD = 0.40
MARGIN_MIN = 0.05
PREDICT_EVERY = 5
SMOOTH_WINDOW = 7
STABLE_NEEDED = 2

# ---------------------------------------------------------- shared state ----
latest = {
    "ok": False, "bin": None, "label": "Starting...", "item": None,
    "note": "", "confidence": 0.0, "phase": "idle", "backend": BACKEND,
}
session = {"counts": {}, "total_items": 0, "est_weight_g": 0.0}
lock = threading.Lock()
stream_lock = threading.Lock()

state = {"phase": "idle"}
last_raw = {"when": None, "message": None, "meta": None, "error": None}

_cache = {}
_cache_lock = threading.Lock()

history = deque(maxlen=SMOOTH_WINDOW)
smoothed_box = None


def set_latest(**kw):
    with lock:
        latest.update(kw)
        latest["phase"] = state["phase"]
        latest["backend"] = BACKEND


def credit_session(bin_name, grams=None):
    if not bin_name:
        return
    with lock:
        session["counts"][bin_name] = session["counts"].get(bin_name, 0) + 1
        session["total_items"] += 1
        session["est_weight_g"] += (
            grams if grams is not None else AVG_WEIGHT_G.get(bin_name, 25))


# ------------------------------------------------------- cloud classifier ---
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "bin": {"type": "string", "enum": ["BIN 1", "BIN 2", "BIN 3", "NONE"]},
        "material": {"type": "string"},
        "estimated_grams": {"type": "number"},
        "confidence": {"type": "number"},
        "note": {"type": "string"},
    },
    "required": ["item", "bin", "material", "estimated_grams",
                 "confidence", "note"],
}

CLASSIFY_PROMPT = (
    "A person is holding one waste item toward the camera. Identify it and "
    "say which bin it belongs in.\n\n"
    "BIN 1 = paper and cardboard\n"
    "BIN 2 = plastics and metals (bottles, cans, foil, tubs)\n"
    "BIN 3 = glass (bottles, jars)\n"
    "NONE  = does not belong in any of these (batteries, food waste, "
    "e-waste, textiles, or mixed items that must be separated first)\n\n"
    "item: 2-4 words naming the object, e.g. 'aluminium drink can'.\n"
    "material: the dominant material.\n"
    "estimated_grams: rough mass of the item in grams.\n"
    "note: one short sentence of disposal advice.\n"
    "confidence: 0.0-1.0. If no item is clearly visible, or you can only "
    "see an empty hand, set confidence to 0.0 and bin to NONE.\n\n"
    "Answer immediately with the JSON object. Do not reason step by step."
)

_ollama = None
_supports_think = True           # flipped False on first TypeError


def ollama_client():
    global _ollama
    if _ollama is None:
        import ollama
        _ollama = ollama.Client(timeout=CLOUD_TIMEOUT_S)
        print(f"ollama client ready (timeout {CLOUD_TIMEOUT_S}s)")
    return _ollama


def chat(messages, fmt=None, num_predict=CLOUD_NUM_PREDICT, tag="call"):
    """
    Single funnel for every model call, with diagnostics attached.

    Returns (content, meta). Raises RuntimeError with a *descriptive* message
    when nothing usable comes back - never a bare JSONDecodeError.
    """
    global _supports_think

    kwargs = dict(
        model=CLOUD_MODEL,
        messages=messages,
        options={"temperature": 0, "num_predict": num_predict},
    )
    if fmt is not None:
        kwargs["format"] = fmt

    t0 = time.time()
    if _supports_think:
        try:
            resp = ollama_client().chat(**kwargs, think=False)
        except TypeError:
            _supports_think = False
            print("  [warn] ollama client has no think= arg; "
                  "run: pip install -U ollama")
            resp = ollama_client().chat(**kwargs)
    else:
        resp = ollama_client().chat(**kwargs)
    elapsed = time.time() - t0

    as_dict = resp if isinstance(resp, dict) else None
    msg = (as_dict or {}).get("message") if as_dict else getattr(resp, "message", {})
    if not isinstance(msg, dict):
        msg = {"content": getattr(msg, "content", None),
               "thinking": getattr(msg, "thinking", None)}

    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()

    def field(name):
        return (as_dict or {}).get(name) if as_dict else getattr(resp, name, None)

    meta = {
        "tag": tag,
        "seconds": round(elapsed, 1),
        "done_reason": field("done_reason"),
        "eval_count": field("eval_count"),
        "prompt_eval_count": field("prompt_eval_count"),
        "content_chars": len(content),
        "thinking_chars": len(thinking),
        "think_arg_used": _supports_think,
        "num_predict": num_predict,
    }

    with lock:
        last_raw.update({
            "when": time.strftime("%H:%M:%S"),
            "message": {"content": content[:4000], "thinking": thinking[:4000]},
            "meta": meta, "error": None,
        })

    if DEBUG_CLOUD:
        print(f"  [{tag}] {elapsed:.1f}s content={len(content)}ch "
              f"thinking={len(thinking)}ch done={meta['done_reason']} "
              f"eval={meta['eval_count']}")
        if not content:
            print(f"  [{tag}] RAW MESSAGE: {json.dumps(msg, default=str)[:600]}")
            if thinking:
                print(f"  [{tag}] THINKING HEAD: {thinking[:300]}")

    if not content:
        if thinking:
            why = (f"model produced {len(thinking)} chars of reasoning and no "
                   f"answer - think=False did not take effect "
                   f"(think arg used: {_supports_think})")
        elif meta["done_reason"] == "length":
            why = (f"output truncated at num_predict={num_predict} before any "
                   f"content - raise CLOUD_NUM_PREDICT")
        else:
            why = "empty message with no reasoning - check /lastraw"
        raise RuntimeError(f"[{tag}] {why}")

    return content, meta


def encode_crop(crop, quality=85):
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def classify_cloud(crop):
    payload = encode_crop(crop)
    key = hashlib.sha1(payload).hexdigest()

    if CACHE_RESPONSES:
        with _cache_lock:
            if key in _cache:
                print("  cloud -> cache hit")
                return _cache[key]

    if DEBUG_CLOUD:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / f"{key[:8]}.jpg").write_bytes(payload)

    content, meta = chat(
        # Gemma 4 performs better with the image before the text.
        messages=[{"role": "user", "content": CLASSIFY_PROMPT,
                   "images": [base64.b64encode(payload).decode()]}],
        fmt=CLASSIFY_SCHEMA,
        tag="classify",
    )

    if DEBUG_CLOUD:
        (DEBUG_DIR / f"{key[:8]}.txt").write_text(content, encoding="utf-8")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        s, t = content.find("{"), content.rfind("}")
        if s != -1 and t > s:
            data = json.loads(content[s:t + 1])
        else:
            raise RuntimeError(
                f"non-JSON reply ({e}). First 200 chars: {content[:200]!r}") from e

    print(f"  cloud -> {data.get('item')} / {data.get('bin')} "
          f"conf={data.get('confidence')} ({meta['seconds']}s)")

    if CACHE_RESPONSES:
        with _cache_lock:
            _cache[key] = data
    return data


def run_classify_cloud(crop):
    try:
        d = classify_cloud(crop)
        conf = float(d.get("confidence", 0))

        if conf < CLOUD_MIN_CONFIDENCE:
            set_latest(ok=False, bin=None, label="Not recognised", item=None,
                       note="Hold the item steady, facing the camera.",
                       confidence=conf)
        elif d["bin"] == "NONE":
            set_latest(ok=False, bin=None,
                       label=f"{d['item']} - none of these bins",
                       item=d["item"], note=d.get("note", ""), confidence=conf)
        else:
            set_latest(ok=True, bin=d["bin"], label=d["item"], item=d["item"],
                       note=d.get("note", ""), confidence=conf)
            credit_session(d["bin"], d.get("estimated_grams"))

    except Exception as e:
        traceback.print_exc()
        with lock:
            last_raw["error"] = f"{type(e).__name__}: {e}"
        set_latest(ok=False, bin=None, label="Cloud error - see console",
                   item=None, note=str(e)[:110], confidence=0.0)
    finally:
        state["phase"] = "result"
        set_latest()


# ---------------------------------------------------------- diagnostics -----
def diag_stages():
    out = []

    # 1 - reach
    try:
        listing = ollama_client().list()
        models = (listing.get("models", []) if isinstance(listing, dict)
                  else listing.models)
        names = []
        for m in models:
            n = ((m.get("model") or m.get("name")) if isinstance(m, dict)
                 else (getattr(m, "model", None) or getattr(m, "name", None)))
            if n:
                names.append(n)
        stage = {"stage": "1 reach", "ok": True,
                 "model_present": CLOUD_MODEL in names, "installed": names[:25]}
        if not stage["model_present"]:
            stage["hint"] = f"run once: ollama pull {CLOUD_MODEL}"
        out.append(stage)
    except Exception as e:
        out.append({"stage": "1 reach", "ok": False, "error": str(e)[:300],
                    "hint": "Ollama not running, or a proxy is intercepting "
                            "localhost:11434 and returning HTML"})
        return out

    # 2 - plain text
    try:
        content, meta = chat([{"role": "user", "content": "Reply with exactly: ok"}],
                             num_predict=64, tag="diag-text")
        out.append({"stage": "2 text", "ok": True, "reply": content[:150],
                    "meta": meta})
    except Exception as e:
        out.append({"stage": "2 text", "ok": False, "error": str(e)[:400],
                    "hint": "no answer even for text - thinking mode is on or "
                            "num_predict is too small"})
        return out

    # 3 - vision
    try:
        img = np.full((240, 240, 3), 255, np.uint8)
        cv2.rectangle(img, (40, 40), (200, 200), (0, 0, 220), -1)
        cv2.circle(img, (120, 120), 45, (255, 255, 255), -1)
        b64 = base64.b64encode(encode_crop(img)).decode()
        content, meta = chat(
            [{"role": "user",
              "content": "Describe the shapes and colours in this image in one "
                         "short sentence.",
              "images": [b64]}],
            num_predict=120, tag="diag-vision")
        understood = any(w in content.lower()
                         for w in ("red", "square", "circle", "white"))
        stage = {"stage": "3 vision", "ok": True, "image_understood": understood,
                 "reply": content[:200], "meta": meta}
        if not understood:
            stage["hint"] = ("image not reaching the model - reply never "
                             "mentions a red square or white circle")
        out.append(stage)
    except Exception as e:
        out.append({"stage": "3 vision", "ok": False, "error": str(e)[:400]})
        return out

    # 4 - schema
    try:
        content, meta = chat(
            [{"role": "user",
              "content": "Return JSON describing a paper coffee cup: item, bin, "
                         "material, estimated_grams, confidence, note."}],
            fmt=CLASSIFY_SCHEMA, tag="diag-schema")
        out.append({"stage": "4 schema", "ok": True,
                    "parsed": json.loads(content), "meta": meta})
    except Exception as e:
        out.append({"stage": "4 schema", "ok": False, "error": str(e)[:400],
                    "hint": "format= schema unsupported - pip install -U ollama"})
    return out


# ------------------------------------------------------- local classifier ---
_local_model = None
_local_labels = None


def local_model():
    global _local_model, _local_labels
    if _local_model is None:
        import tf_keras
        _local_model = tf_keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", encoding="utf-8") as f:
            _local_labels = [ln.strip().split(" ", 1)[1] for ln in f if ln.strip()]
        print(f"local model loaded: {_local_labels}")
    return _local_model, _local_labels


def preprocess_bgr(crop):
    from PIL import Image
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


def classify_local(crop):
    model, labels = local_model()
    scores = model.predict(preprocess_bgr(crop), verbose=0)[0]
    order = np.argsort(scores)[::-1]
    top, second = float(scores[order[0]]), float(scores[order[1]])
    return labels[int(order[0])], top, top - second


# ------------------------------------------------------------- geometry -----
def roi_box(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * ROI_FRACTION)
    x1, y1 = (w - side) // 2, (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def in_zone(cx, cy, fw, fh):
    zx1, zy1 = fw * (1 - ZONE_FRACTION) / 2, fh * (1 - ZONE_FRACTION) / 2
    return zx1 <= cx <= fw - zx1 and zy1 <= cy <= fh - zy1


def zone_rect(fw, fh):
    zx1 = int(fw * (1 - ZONE_FRACTION) / 2)
    zy1 = int(fh * (1 - ZONE_FRACTION) / 2)
    return zx1, zy1, fw - zx1, fh - zy1


def palm_has_object(crop):
    h, w = crop.shape[:2]
    centre = crop[int(h * .25):int(h * .75), int(w * .25):int(w * .75)]
    if centre.size == 0:
        return 0.0
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 170, 255]))
    return 1.0 - (np.count_nonzero(skin) / skin.size)


# --------------------------------------------------------- hand tracking ----
hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)


def hand_roi_and_grasp(frame, timestamp_ms, prev_center):
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
    for lms in res.hand_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms])
        cx, cy = pts.mean(axis=0)
        if not in_zone(cx, cy, w, h):
            continue
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        wrist = lms[0]
        tip_d = np.mean([math.hypot(lms[i].x - wrist.x, lms[i].y - wrist.y)
                         for i in (4, 8, 12, 16, 20)])
        palm_d = np.mean([math.hypot(lms[i].x - wrist.x, lms[i].y - wrist.y)
                          for i in (2, 5, 9, 13, 17)])
        candidates.append({"cx": cx, "cy": cy, "span": span,
                           "grasping": tip_d < palm_d * GRASP_RATIO})

    if not candidates:
        smoothed_box = None
        return None, False

    if prev_center is not None:
        chosen = min(candidates, key=lambda c: math.hypot(
            c["cx"] - prev_center[0], c["cy"] - prev_center[1]))
    else:
        grasping = [c for c in candidates if c["grasping"]]
        chosen = max(grasping or candidates, key=lambda c: c["span"])

    side = max(chosen["span"] * BOX_SCALE, MIN_BOX_SIDE)
    box = np.array([chosen["cx"] - side / 2, chosen["cy"] - side / 2,
                    chosen["cx"] + side / 2, chosen["cy"] + side / 2])
    smoothed_box = box if smoothed_box is None else (
        SMOOTH_ALPHA * box + (1 - SMOOTH_ALPHA) * smoothed_box)

    x1, y1, x2, y2 = smoothed_box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None, False
    return (x1, y1, x2, y2), chosen["grasping"]


# ------------------------------------------------------------ main loop -----
def generate_frames():
    if not stream_lock.acquire(blocking=False):
        print("stream already active - refusing second connection")
        return

    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("camera opened:", cap.isOpened())
        for _ in range(10):
            cap.read()

        frame_no = 0
        fallback_box = None
        last_box, last_grasping, last_center = None, False, None
        nothing_since = None

        steady, steady_center = 0, None
        last_call_at = 0.0
        freeze_img, freeze_box = None, None

        stable_count, locked_label, last_conf = 0, None, 0.0

        state["phase"] = "idle"
        set_latest(label="Show me your hand", ok=False, bin=None)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()

            if fallback_box is None:
                fallback_box = roi_box(frame)

            if frame_no % DETECT_EVERY == 0:
                last_box, last_grasping = hand_roi_and_grasp(
                    frame, int(now * 1000), last_center)
                last_center = None
                if last_box is not None:
                    last_center = ((last_box[0] + last_box[2]) / 2,
                                   (last_box[1] + last_box[3]) / 2)

            tracking = last_box is not None
            x1, y1, x2, y2 = last_box if tracking else fallback_box

            if tracking:
                nothing_since = None
            elif nothing_since is None:
                nothing_since = now

            if BACKEND == "cloud":
                if state["phase"] == "idle" and tracking and last_grasping:
                    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
                    if steady_center and math.dist(centre, steady_center) < STEADY_PX:
                        steady += 1
                    else:
                        steady = 0
                    steady_center = centre

                    if steady >= STEADY_FRAMES and now - last_call_at > MIN_GAP_S:
                        fill = palm_has_object(frame[y1:y2, x1:x2])
                        if fill >= OBJECT_FILL_MIN:
                            print(f"trigger: steady={steady} fill={fill:.2f}")
                            state["phase"] = "classifying"
                            last_call_at = now
                            freeze_img = frame.copy()
                            freeze_box = (x1, y1, x2, y2)
                            set_latest(ok=False, bin=None, label="Analysing...",
                                       item=None, note="", confidence=0.0)
                            threading.Thread(
                                target=run_classify_cloud,
                                args=(frame[y1:y2, x1:x2].copy(),),
                                daemon=True).start()
                        elif frame_no % 30 == 0:
                            print(f"  waiting: fill={fill:.2f} "
                                  f"(need {OBJECT_FILL_MIN})")

                elif state["phase"] == "result":
                    if nothing_since and now - nothing_since >= RESET_AFTER_S:
                        state["phase"] = "idle"
                        steady, steady_center = 0, None
                        freeze_img, freeze_box = None, None
                        set_latest(ok=False, bin=None, label="Show me your hand",
                                   item=None, note="", confidence=0.0)

                if state["phase"] == "idle" and not tracking:
                    steady, steady_center = 0, None

            else:
                if not tracking:
                    history.append(NOTHING_LABEL)
                    if nothing_since and now - nothing_since >= RESET_AFTER_S:
                        locked_label, stable_count = None, 0
                elif locked_label is not None:
                    pass
                elif not last_grasping:
                    history.append(NOTHING_LABEL)
                elif palm_has_object(frame[y1:y2, x1:x2]) < OBJECT_FILL_MIN:
                    history.append(NOTHING_LABEL)
                elif frame_no % PREDICT_EVERY == 0:
                    label, conf, margin = classify_local(frame[y1:y2, x1:x2])
                    last_conf = conf
                    good = conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_MIN
                    history.append(label if good else NOTHING_LABEL)

                if locked_label is None and history:
                    voted = Counter(history).most_common(1)[0][0]
                    if (tracking and last_grasping and voted != NOTHING_LABEL
                            and voted == history[-1]):
                        stable_count += 1
                    else:
                        stable_count = 0

                    if stable_count >= STABLE_NEEDED:
                        locked_label = voted
                        freeze_img = frame.copy()
                        freeze_box = (x1, y1, x2, y2)
                        bin_name = LOCAL_BIN.get(voted)
                        state["phase"] = "result"
                        set_latest(ok=bin_name is not None, bin=bin_name,
                                   label=BIN_MAP.get(bin_name, (voted, GREY))[0],
                                   item=voted, note="", confidence=last_conf)
                        credit_session(bin_name)
                    else:
                        state["phase"] = "idle"
                        set_latest(ok=False, bin=None,
                                   label="Show me your hand" if not tracking
                                   else "Hold an item to scan")

            frame_no += 1

            # ------------------------------------------------------ draw ---
            with lock:
                phase = state["phase"]
                cap_bin, cap_label = latest["bin"], latest["label"]
                cap_note = latest["note"]

            colour = BIN_MAP.get(cap_bin, (None, GREY))[1] if cap_bin else GREY

            if phase in ("classifying", "result") and freeze_img is not None:
                display = freeze_img.copy()
                bx1, by1, bx2, by2 = freeze_box
            else:
                display = frame
                bx1, by1, bx2, by2 = x1, y1, x2, y2

            if phase == "classifying":
                t = (math.sin(now * 6) + 1) / 2
                colour = tuple(int(60 + 120 * t) for _ in range(3))
                headline = "Analysing..."
            elif phase == "result":
                headline = f"{cap_bin}  {chr(0x2713)}" if cap_bin else cap_label
            elif not tracking:
                headline = "Show me your hand"
            elif not last_grasping:
                headline = "Hold an item to scan"
            else:
                headline = "Hold steady..."

            cv2.rectangle(display, (bx1, by1), (bx2, by2), colour,
                          4 if phase != "idle" else 2)
            cv2.putText(display, headline, (bx1, max(24, by1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2, cv2.LINE_AA)

            if phase == "result" and cap_bin:
                cv2.putText(display, cap_label,
                            (bx1, min(display.shape[0] - 30, by2 + 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
            if phase == "result" and cap_note:
                cv2.putText(display, cap_note[:52], (12, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220),
                            1, cv2.LINE_AA)

            zx1, zy1, zx2, zy2 = zone_rect(frame.shape[1], frame.shape[0])
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (70, 70, 75), 1)

            _, buf = cv2.imencode(".jpg", display)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

    finally:
        if cap is not None:
            cap.release()
        stream_lock.release()


# --------------------------------------------------------------- routes -----
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


@app.route("/backend/<name>", methods=["POST"])
def set_backend(name):
    global BACKEND
    if name not in ("cloud", "local"):
        return jsonify({"ok": False, "error": "cloud or local"}), 400
    BACKEND = name
    state["phase"] = "idle"
    history.clear()
    set_latest(ok=False, bin=None, label="Show me your hand",
               item=None, note="", confidence=0.0)
    print(f"backend -> {BACKEND}")
    return jsonify({"ok": True, "backend": BACKEND})


@app.route("/diag")
def diag():
    stages = diag_stages()
    bad = next((s for s in stages
                if not s.get("ok")
                or s.get("model_present") is False
                or s.get("image_understood") is False), None)
    return jsonify({
        "model": CLOUD_MODEL,
        "num_predict": CLOUD_NUM_PREDICT,
        "think_arg_supported": _supports_think,
        "verdict": bad["stage"] if bad else "all passed",
        "hint": (bad or {}).get("hint"),
        "stages": stages,
    })


@app.route("/lastraw")
def lastraw():
    with lock:
        return jsonify(dict(last_raw))


@app.route("/health")
def health():
    if BACKEND != "cloud":
        return jsonify({"backend": BACKEND, "cloud_reachable": None})
    try:
        content, meta = chat([{"role": "user", "content": "Reply with: ok"}],
                             num_predict=32, tag="health")
        return jsonify({"backend": BACKEND, "cloud_reachable": True,
                        "model": CLOUD_MODEL, "reply": content[:80],
                        "meta": meta})
    except Exception as e:
        return jsonify({"backend": BACKEND, "cloud_reachable": False,
                        "error": str(e)[:300]}), 503


if __name__ == "__main__":
    print(f"backend={BACKEND}  debug={DEBUG_CLOUD}  model={CLOUD_MODEL}")
    print("when the cloud path fails, open:  http://127.0.0.1:5000/diag")
    if BACKEND == "local":
        local_model()
    app.run(debug=True, use_reloader=False)