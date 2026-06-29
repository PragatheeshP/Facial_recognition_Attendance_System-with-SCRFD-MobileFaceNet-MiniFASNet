"""
Thin wrapper around InsightFace's lightweight "buffalo_s" model pack.

buffalo_s = SCRFD-500MF (face detection, ~2.5MB) + MobileFaceNet
(512-d face embedding, ~4MB). Both run happily on CPU via ONNXRuntime —
no CUDA, no multi-hundred-MB ResNet backbones.

First run downloads the pack automatically (~13MB) into ~/.insightface/.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import config


@dataclass
class DetectedFace:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    det_score: float
    embedding: np.ndarray  # L2-normalized, shape (512,)
    face_area_ratio: float  # face_area / frame_area, 0.0-1.0


class FaceEngine:
    def __init__(self):
        # Imported lazily so the rest of the app can be explored/tested
        # without insightface installed.
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_PACK,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=config.DETECTION_SIZE)

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Runs detection + embedding in one pass.
        image_bgr: numpy array as returned by cv2.imread / cv2 frame capture.
        """
        h, w = image_bgr.shape[:2]
        frame_area = h * w
        faces = self._app.get(image_bgr)
        results = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            face_area = float((x2 - x1) * (y2 - y1))
            face_area_ratio = face_area / frame_area if frame_area > 0 else 0.0
            results.append(
                DetectedFace(
                    bbox=face.bbox,
                    det_score=float(face.det_score),
                    embedding=face.normed_embedding.astype("float32"),
                    face_area_ratio=face_area_ratio,
                )
            )
        return results

    def best_single_face(self, image_bgr: np.ndarray) -> Optional[DetectedFace]:
        """Used during enrollment, where each capture should contain exactly one face."""
        faces = self.detect_and_embed(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.det_score)