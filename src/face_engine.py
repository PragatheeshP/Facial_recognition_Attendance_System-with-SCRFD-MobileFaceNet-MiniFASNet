"""
Thin wrapper around InsightFace's detection + recognition models, loaded
directly from local .onnx files in resources/face_models/ instead of
FaceAnalysis's auto-downloader.

Why: FaceAnalysis(name="buffalo_s") always downloads the FULL pack zip
(~124MB, 5 models: detection, recognition, genderage, 2 landmark models)
even if you only ask it to load 2 of them via allowed_modules — the
filtering happens AFTER download/extraction, not before. That download
was what was blowing past Render's 512MB limit at startup. Loading just
the 2 committed .onnx files we actually use avoids any network call.
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
        import os
        from insightface.model_zoo import get_model

        det_path = os.path.join(config.FACE_MODEL_DIR, "det_500m.onnx")
        rec_path = os.path.join(config.FACE_MODEL_DIR, "w600k_mbf.onnx")

        self._det_model = get_model(det_path, providers=["CPUExecutionProvider"])
        self._det_model.prepare(ctx_id=0, input_size=config.DETECTION_SIZE)

        self._rec_model = get_model(rec_path, providers=["CPUExecutionProvider"])
        self._rec_model.prepare(ctx_id=0)

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        from insightface.utils import face_align

        h, w = image_bgr.shape[:2]
        frame_area = h * w

        bboxes, kpss = self._det_model.detect(image_bgr, input_size=config.DETECTION_SIZE)

        results = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, det_score = bboxes[i]
            kps = kpss[i] if kpss is not None else None
            if kps is None:
                continue  # can't align/embed without landmarks

            aligned = face_align.norm_crop(
                image_bgr, landmark=kps, image_size=self._rec_model.input_size[0]
            )
            raw_embedding = self._rec_model.get_feat(aligned).flatten()
            normed_embedding = (raw_embedding / np.linalg.norm(raw_embedding)).astype("float32")

            face_area = float((x2 - x1) * (y2 - y1))
            face_area_ratio = face_area / frame_area if frame_area > 0 else 0.0

            results.append(
                DetectedFace(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    det_score=float(det_score),
                    embedding=normed_embedding,
                    face_area_ratio=face_area_ratio,
                )
            )
        return results