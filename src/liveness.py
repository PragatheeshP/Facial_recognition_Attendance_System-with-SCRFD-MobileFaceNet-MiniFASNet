"""
Liveness / anti-spoofing — MiniFASNet (Silent-Face-Anti-Spoofing).

This replaces the old blur-heuristic placeholder with the real model the
project always intended to use: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

What it does
------------
MiniFASNet looks at the texture/frequency artifacts that give away a
*presentation attack* — a printed photo or a phone/tablet screen held up to
the camera instead of an actual face. Things like screen moire patterns,
paper grain, and the way printers/displays reproduce skin tone all leave a
signature that a live face doesn't have. It does NOT need motion, blinking,
or multiple frames — it works on a single still image, which matches how
this app captures frames (one `st.camera_input` snapshot at a time).

The official project ships two tiny model variants that each look at the
face crop at a different amount of surrounding context (a tighter crop and
a wider one, expressed as a multiple of the detected face box). Their
predictions are summed and averaged — this two-scale ensemble is part of
why the original model is robust on both close-up and slightly-back photos.

We run the *original* pretrained weights, converted from the upstream
.pth checkpoints to ONNX (see convert_to_onnx.py for the one-time
conversion script) so this module only depends on onnxruntime at runtime —
no torch needed in production, same pattern as face_engine.py.

Important: this expects the FULL captured frame plus the face bounding box
from face_engine, not an already-cropped face. The model needs to see some
of the surrounding context (e.g. a phone bezel, a sheet of paper's edge) to
do its job — a tight crop throws that signal away.
"""

import os
from typing import List, Tuple

import numpy as np

from . import config

_sessions = None  # lazily-initialized list of (onnxruntime.InferenceSession, crop_scale, h, w)


def _get_sessions():
    global _sessions
    if _sessions is None:
        import onnxruntime as ort

        sessions = []
        for filename, scale in config.ANTI_SPOOF_MODELS:
            path = os.path.join(config.ANTI_SPOOF_MODEL_DIR, filename)
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            # input shape is fixed at export time: (1, 3, H, W)
            h, w = sess.get_inputs()[0].shape[2], sess.get_inputs()[0].shape[3]
            sessions.append((sess, scale, h, w))
        _sessions = sessions
    return _sessions


def _expanded_crop(frame_bgr: np.ndarray, bbox_xyxy, scale: float, out_h: int, out_w: int) -> np.ndarray:
    """
    Re-implementation of the official repo's CropImage._get_new_box + crop:
    take a box `scale`x the size of the detected face, centered on the face,
    clamped to the frame, then resized to (out_h, out_w).
    """
    src_h, src_w = frame_bgr.shape[0], frame_bgr.shape[1]
    x1, y1, x2, y2 = bbox_xyxy
    box_w, box_h = x2 - x1, y2 - y1

    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x1 + box_w / 2, y1 + box_h / 2

    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1

    left, top, right, bottom = int(left), int(top), int(right), int(bottom)

    import cv2

    patch = frame_bgr[top:bottom + 1, left:right + 1]
    return cv2.resize(patch, (out_w, out_h))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def liveness_score(frame_bgr: np.ndarray, bbox_xyxy) -> float:
    """
    Returns the averaged probability (0.0-1.0) that the face at bbox_xyxy in
    frame_bgr is a real, live face rather than a printed photo or
    screen replay. Class index 1 = "real" in the original model's 3-class
    output (0 and 2 are different categories of spoof).
    """
    total = np.zeros(3, dtype=np.float32)
    for sess, scale, h, w in _get_sessions():
        crop = _expanded_crop(frame_bgr, bbox_xyxy, scale, h, w)
        # Matches the original repo's preprocessing exactly: BGR, HWC->CHW,
        # cast to float32, NO /255 scaling and NO mean/std normalization —
        # that one detail is easy to get wrong and silently breaks accuracy.
        inp = crop.transpose(2, 0, 1).astype(np.float32)[None, ...]
        logits = sess.run(None, {"input": inp})[0][0]
        total += _softmax(logits)

    real_prob = float(total[1] / len(config.ANTI_SPOOF_MODELS))
    return real_prob


def check_liveness(frame_bgr: np.ndarray, bbox_xyxy) -> bool:
    """
    Returns True if the face should be treated as live.

    frame_bgr: the full captured frame (NOT a pre-cropped face) — needed so
               the model can see context around the face box.
    bbox_xyxy: [x1, y1, x2, y2] from face_engine.DetectedFace.bbox.
    """
    return liveness_result(frame_bgr, bbox_xyxy)["is_live"]


def liveness_result(frame_bgr: np.ndarray, bbox_xyxy) -> dict:
    """
    Same check as check_liveness(), but also returns the score and a
    human-readable reason — this is what attendance_service.py uses so a
    rejection can be explained in the UI instead of just a bare status.

    Returns: {"is_live": bool, "score": float, "reason": str}
    """
    if not config.ENABLE_LIVENESS:
        return {"is_live": True, "score": 1.0, "reason": "Liveness check disabled."}

    score = liveness_score(frame_bgr, bbox_xyxy)
    is_live = score >= config.LIVENESS_THRESHOLD
    reason = (
        f"Live face (confidence {score:.0%})."
        if is_live
        else _spoof_reason(score)
    )
    return {"is_live": is_live, "score": score, "reason": reason}


def _spoof_reason(score: float) -> str:
    """
    Builds the human-readable explanation for a rejected face.

    Honest limitation: the model's 3-class output doesn't reliably tell us
    *which* kind of spoof it saw — class 0 and class 2 are both "not real"
    but the official project doesn't document a clean print-vs-screen
    mapping between them, and this model was trained mainly on printed
    photos and screen replays (not 3D masks). So rather than guess an
    attack type we don't have, the message reports the confidence and the
    most likely real-world causes.
    """
    if score < 0.15:
        return (
            f"Spoof rejected \u2014 very low liveness confidence ({score:.0%}). "
            f"Strong sign of a printed photo or a phone/screen shown to the camera."
        )
    return (
        f"Spoof rejected \u2014 liveness confidence {score:.0%} is below the "
        f"{config.LIVENESS_THRESHOLD:.0%} threshold. Could be a spoof attempt, or just "
        f"a poor-quality capture (blur, harsh lighting, extreme angle) \u2014 try recapturing."
    )
