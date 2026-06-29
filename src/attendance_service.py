"""
Business logic layer. This is the only module the Streamlit UI talks to —
it owns the rules (dedup window, threshold, re-enrollment flagging,
proximity rejection, and periodic presence verification) so the UI stays
dumb and the rules stay testable on their own.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .db import Student, ClassSession, AttendanceRecord, get_db
from .face_engine import FaceEngine
from .vector_store import VectorStore
from .liveness import liveness_result


class AttendanceService:
    def __init__(self):
        self.face_engine = FaceEngine()
        self.vector_store = VectorStore()

    # ---------- Enrollment ----------

    def enroll_student(
        self,
        student_code: str,
        name: str,
        class_id: str,
        images_bgr: List[np.ndarray],
    ) -> str:
        """
        Takes several photos of one student (frontal, slight-left,
        slight-right, etc.), averages their embeddings into a single
        reference vector, and stores it.
        Returns a human-readable status message.
        """
        embeddings = []
        for img in images_bgr:
            face = self.face_engine.best_single_face(img)
            if face is None:
                continue
            embeddings.append(face.embedding)

        if not embeddings:
            return "No usable face found in any of the captured images — try again with better lighting."

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        db = get_db()
        try:
            student = db.query(Student).filter_by(student_code=student_code).first()
            if student is None:
                student = Student(student_code=student_code, name=name, class_id=class_id)
                db.add(student)
                db.commit()
                db.refresh(student)
            else:
                student.name = name
                student.class_id = class_id
                db.commit()

            self.vector_store.upsert(student.id, avg_embedding.astype("float32"))
            return f"Enrolled {name} ({student_code}) using {len(embeddings)}/{len(images_bgr)} usable images."
        finally:
            db.close()

    # ---------- Sessions ----------

    def start_session(self, classroom_id: str, subject: str, faculty: str) -> int:
        db = get_db()
        try:
            session = ClassSession(classroom_id=classroom_id, subject=subject, faculty=faculty)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
        finally:
            db.close()

    def end_session(self, session_id: int):
        db = get_db()
        try:
            session = db.get(ClassSession, session_id)
            if session:
                session.status = "COMPLETED"
                session.ended_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()

    # ---------- Recognition + attendance ----------

    def process_frame(
        self,
        session_id: int,
        frame_bgr: np.ndarray,
        snapshot_number: int = 0,
        is_periodic: bool = False,
    ) -> List[dict]:
        """
        Runs detection + recognition on one captured frame and marks
        attendance for every student recognized above threshold.

        Args:
            session_id: active class session
            frame_bgr: captured frame
            snapshot_number: 0 for manual capture, 1+ for periodic auto-checks
            is_periodic: whether this call came from the automatic timer

        Returns a list of per-face result dicts for display in the UI.
        """
        faces = self.face_engine.detect_and_embed(frame_bgr)
        results = []
        detected_student_ids = set()

        db = get_db()
        try:
            for face in faces:
                # ---- 1. Proximity / frame-fill check ----
                if face.face_area_ratio > config.MAX_FACE_FILL_RATIO:
                    results.append(
                        {
                            "bbox": face.bbox,
                            "status": "TOO_CLOSE",
                            "name": None,
                            "confidence": round(face.face_area_ratio, 3),
                            "reason": (
                                f"Face fills {face.face_area_ratio:.0%} of the frame "
                                f"(max allowed: {config.MAX_FACE_FILL_RATIO:.0%}). "
                                f"Please step back so the camera can see more context."
                            ),
                        }
                    )
                    self._record_attendance(
                        db, session_id, None, "TOO_CLOSE", face.face_area_ratio,
                        snapshot_number=snapshot_number, is_periodic=is_periodic
                    )
                    continue

                # ---- 2. Liveness / anti-spoof check ----
                liveness = liveness_result(frame_bgr, face.bbox)
                if not liveness["is_live"]:
                    results.append(
                        {
                            "bbox": face.bbox,
                            "status": "SPOOF_REJECTED",
                            "name": None,
                            "confidence": round(liveness["score"], 3),
                            "reason": liveness["reason"],
                        }
                    )
                    self._record_attendance(
                        db, session_id, None, "SPOOF_REJECTED", liveness["score"],
                        snapshot_number=snapshot_number, is_periodic=is_periodic
                    )
                    continue

                # ---- 3. Face recognition ----
                matches = self.vector_store.search(face.embedding, top_k=1)
                if not matches or matches[0][1] < config.RECOGNITION_THRESHOLD:
                    results.append(
                        {
                            "bbox": face.bbox,
                            "status": "UNRECOGNIZED",
                            "name": None,
                            "confidence": None,
                            "reason": None,
                        }
                    )
                    self._record_attendance(
                        db, session_id, None, "UNRECOGNIZED", None,
                        snapshot_number=snapshot_number, is_periodic=is_periodic
                    )
                    continue

                student_id, score = matches[0]
                student = db.get(Student, student_id)
                detected_student_ids.add(student_id)

                # ---- 4. Dedup check ----
                already_marked = self._already_marked_recently(db, session_id, student_id)
                if already_marked:
                    results.append(
                        {
                            "bbox": face.bbox,
                            "status": "ALREADY_MARKED",
                            "name": student.name,
                            "confidence": score,
                            "reason": None,
                        }
                    )
                    continue

                # ---- 5. Mark present ----
                self._record_attendance(
                    db, session_id, student_id, "PRESENT", score,
                    snapshot_number=snapshot_number, is_periodic=is_periodic
                )
                self._update_rolling_confidence(db, student, score)
                results.append(
                    {
                        "bbox": face.bbox,
                        "status": "PRESENT",
                        "name": student.name,
                        "confidence": score,
                        "reason": None,
                    }
                )

            # ---- 6. Periodic check: flag students who LEFT ----
            # If this is a periodic check, find students previously marked PRESENT
            # who are NOT in the current frame and mark them as LEFT.
            if is_periodic and config.PERIODIC_CHECK_INTERVAL_MINUTES > 0:
                self._flag_absent_students(db, session_id, detected_student_ids, snapshot_number)

        finally:
            db.close()

        return results

    def _already_marked_recently(self, db, session_id: int, student_id: int) -> bool:
        """
        Within a single session we only want one PRESENT mark per student,
        regardless of how many snapshots are taken.
        """
        existing = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.session_id == session_id,
                AttendanceRecord.student_id == student_id,
                AttendanceRecord.status == "PRESENT",
            )
            .first()
        )
        return existing is not None

    def _record_attendance(
        self,
        db,
        session_id,
        student_id,
        status,
        confidence,
        snapshot_number: int = 0,
        is_periodic: bool = False,
    ):
        record = AttendanceRecord(
            session_id=session_id,
            student_id=student_id,
            status=status,
            confidence=confidence,
            snapshot_number=snapshot_number,
            is_periodic=int(is_periodic),
        )
        db.add(record)
        db.commit()

    def _update_rolling_confidence(self, db, student: Student, latest_score: float):
        if student.last_avg_confidence is None:
            student.last_avg_confidence = latest_score
        else:
            # simple exponential smoothing rather than storing full history
            student.last_avg_confidence = 0.7 * student.last_avg_confidence + 0.3 * latest_score
        student.needs_reenrollment = int(student.last_avg_confidence < config.RE_ENROLL_ACCURACY_THRESHOLD)
        db.commit()

    def _flag_absent_students(
        self,
        db,
        session_id: int,
        detected_student_ids: set,
        snapshot_number: int,
    ):
        """
        During a periodic check, mark any previously-present student who is
        no longer visible as LEFT.  We only do this once per student per
        session to avoid spamming the record table.
        """
        # Find all students ever marked PRESENT in this session
        present_records = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.session_id == session_id,
                AttendanceRecord.status == "PRESENT",
            )
            .all()
        )

        present_student_ids = {r.student_id for r in present_records}

        # Find students already flagged as LEFT (so we don't duplicate)
        left_records = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.session_id == session_id,
                AttendanceRecord.status == "LEFT",
            )
            .all()
        )
        already_left_ids = {r.student_id for r in left_records}

        for student_id in present_student_ids:
            if student_id not in detected_student_ids and student_id not in already_left_ids:
                self._record_attendance(
                    db, session_id, student_id, "LEFT", None,
                    snapshot_number=snapshot_number, is_periodic=True
                )

    # ---------- Periodic check helpers ----------

    def should_run_periodic_check(self, session_id: int, last_check_time: Optional[datetime]) -> bool:
        """
        Returns True if enough time has elapsed since the last periodic check
        to warrant a new one.
        """
        if config.PERIODIC_CHECK_INTERVAL_MINUTES <= 0:
            return False
        if last_check_time is None:
            return True
        elapsed = datetime.now(timezone.utc) - last_check_time
        return elapsed >= timedelta(minutes=config.PERIODIC_CHECK_INTERVAL_MINUTES)

    def get_still_present(self, session_id: int) -> List[dict]:
        """
        Returns students who are still considered present in the session,
        i.e. they were marked PRESENT and have NOT been marked LEFT.
        """
        db = get_db()
        try:
            # All PRESENT records in this session
            present_records = (
                db.query(AttendanceRecord)
                .filter(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.status == "PRESENT",
                )
                .all()
            )
            present_ids = {r.student_id for r in present_records}

            # All LEFT records in this session
            left_records = (
                db.query(AttendanceRecord)
                .filter(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.status == "LEFT",
                )
                .all()
            )
            left_ids = {r.student_id for r in left_records}

            still_present_ids = present_ids - left_ids
            students = []
            for sid in still_present_ids:
                student = db.get(Student, sid)
                if student:
                    # Get the latest PRESENT record for this student in this session
                    latest = (
                        db.query(AttendanceRecord)
                        .filter(
                            AttendanceRecord.session_id == session_id,
                            AttendanceRecord.student_id == sid,
                            AttendanceRecord.status == "PRESENT",
                        )
                        .order_by(AttendanceRecord.marked_at.desc())
                        .first()
                    )
                    students.append({
                        "id": student.id,
                        "student_code": student.student_code,
                        "name": student.name,
                        "last_seen_at": latest.marked_at if latest else None,
                    })
            return students
        finally:
            db.close()

    def get_left_students(self, session_id: int) -> List[dict]:
        """Returns students who were marked PRESENT but later flagged as LEFT."""
        db = get_db()
        try:
            left_records = (
                db.query(AttendanceRecord)
                .filter(
                    AttendanceRecord.session_id == session_id,
                    AttendanceRecord.status == "LEFT",
                )
                .all()
            )
            result = []
            for r in left_records:
                if r.student:
                    result.append({
                        "id": r.student.id,
                        "student_code": r.student.student_code,
                        "name": r.student.name,
                        "left_at": r.marked_at,
                        "snapshot_number": r.snapshot_number,
                    })
            return result
        finally:
            db.close()

    # ---------- Reporting ----------

    def get_session_attendance(self, session_id: int) -> List[dict]:
        db = get_db()
        try:
            records = (
                db.query(AttendanceRecord)
                .filter(AttendanceRecord.session_id == session_id)
                .order_by(AttendanceRecord.marked_at)
                .all()
            )
            rows = []
            for r in records:
                rows.append(
                    {
                        "student": r.student.name if r.student else "Unrecognized",
                        "status": r.status,
                        "confidence": round(r.confidence, 3) if r.confidence else None,
                        "marked_at": r.marked_at,
                        "snapshot": r.snapshot_number,
                        "auto": bool(r.is_periodic),
                    }
                )
            return rows
        finally:
            db.close()

    def list_students(self) -> List[dict]:
        db = get_db()
        try:
            students = db.query(Student).all()
            return [
                {
                    "id": s.id,
                    "student_code": s.student_code,
                    "name": s.name,
                    "class_id": s.class_id,
                    "needs_reenrollment": bool(s.needs_reenrollment),
                }
                for s in students
            ]
        finally:
            db.close()

    def list_sessions(self) -> List[dict]:
        db = get_db()
        try:
            sessions = db.query(ClassSession).order_by(ClassSession.started_at.desc()).all()
            return [
                {
                    "id": s.id,
                    "classroom_id": s.classroom_id,
                    "subject": s.subject,
                    "faculty": s.faculty,
                    "started_at": s.started_at,
                    "status": s.status,
                }
                for s in sessions
            ]
        finally:
            db.close()