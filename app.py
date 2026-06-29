"""
Flask-based Classroom Attendance System
Web deployment-ready replacement for the PyQt5 desktop app.

Features preserved from original:
- Student enrollment with photo capture
- Live attendance sessions with face recognition
- Periodic auto-checks (server-side timer + browser notification)
- Reports and CSV export
- Anti-spoofing liveness detection
- Dashboard showing present/left students
"""

import os
import base64
import io
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from src.db import init_db
from src.attendance_service import AttendanceService

# Initialize database
init_db()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# SocketIO for real-time updates (optional but useful for periodic checks)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global service instance
service = AttendanceService()

# Thread-safe storage for periodic check timers
active_timers = {}  # session_id -> threading.Timer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(data_url):
    """Convert base64 data URL to OpenCV BGR image."""
    if ',' in data_url:
        header, encoded = data_url.split(',', 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def encode_image(img_bgr):
    """Convert OpenCV BGR image to base64 PNG data URL."""
    _, buffer = cv2.imencode('.png', img_bgr)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


def annotate_frame(frame_bgr, results):
    """Draw bounding boxes and labels on frame (same logic as PyQt5 app)."""
    annotated = frame_bgr.copy()
    color_map = {
        "PRESENT": (0, 200, 0),
        "ALREADY_MARKED": (200, 150, 0),
        "UNRECOGNIZED": (0, 0, 200),
        "SPOOF_REJECTED": (0, 0, 120),
        "TOO_CLOSE": (200, 0, 200),
        "LEFT": (128, 128, 128),
    }

    def wrap_text(text, max_chars=32):
        words = text.split()
        lines, current = [], ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars:
                current = f"{current} {word}".strip()
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        color = color_map.get(r["status"], (128, 128, 128))
        label = r.get("name") or r["status"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if r.get("reason") and r["status"] in ("SPOOF_REJECTED", "TOO_CLOSE"):
            for i, line in enumerate(wrap_text(r["reason"], max_chars=36)[:3]):
                cv2.putText(annotated, line, (x1, y2 + 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return annotated


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Home page with navigation."""
    return render_template('index.html')


@app.route('/enroll')
def enroll_page():
    """Student enrollment page."""
    students = service.list_students()
    return render_template('enroll.html', students=students, 
                          num_required=config.NUM_ENROLLMENT_IMAGES)


@app.route('/attendance')
def attendance_page():
    """Attendance taking page."""
    sessions = service.list_sessions()
    active_sessions = [s for s in sessions if s['status'] == 'ACTIVE']
    return render_template('attendance.html', 
                          active_sessions=active_sessions,
                          periodic_interval=config.PERIODIC_CHECK_INTERVAL_MINUTES)


@app.route('/reports')
def reports_page():
    """Reports page."""
    sessions = service.list_sessions()
    return render_template('reports.html', sessions=sessions)


# ---------------------------------------------------------------------------
# API Routes - Enrollment
# ---------------------------------------------------------------------------

@app.route('/api/students', methods=['GET'])
def list_students():
    """Get all enrolled students."""
    return jsonify(service.list_students())


@app.route('/api/enroll', methods=['POST'])
def enroll_student():
    """Enroll a new student with captured images."""
    data = request.get_json()

    student_code = data.get('student_code', '').strip()
    name = data.get('name', '').strip()
    class_id = data.get('class_id', '').strip()
    images_data = data.get('images', [])

    if not student_code or not name:
        return jsonify({"success": False, "error": "Student code and name are required."}), 400

    if len(images_data) < 3:
        return jsonify({"success": False, "error": f"Need at least 3 photos. Got {len(images_data)}."}), 400

    # Decode images
    images_bgr = []
    for img_data in images_data:
        img = decode_image(img_data)
        if img is not None:
            images_bgr.append(img)

    if len(images_bgr) < 3:
        return jsonify({"success": False, "error": "Could not decode at least 3 valid images."}), 400

    try:
        msg = service.enroll_student(student_code, name, class_id, images_bgr)
        return jsonify({"success": True, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/student/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a student (optional feature)."""
    # Note: This would need to be added to AttendanceService
    # For now, return not implemented
    return jsonify({"success": False, "error": "Delete not implemented in this version."}), 501


# ---------------------------------------------------------------------------
# API Routes - Sessions
# ---------------------------------------------------------------------------

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Get all sessions."""
    return jsonify(service.list_sessions())


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Start a new attendance session."""
    data = request.get_json()
    classroom_id = data.get('classroom_id', 'ROOM-1').strip()
    subject = data.get('subject', '').strip()
    faculty = data.get('faculty', '').strip()

    try:
        session_id = service.start_session(classroom_id, subject, faculty)

        # Start periodic check timer if enabled
        if config.PERIODIC_CHECK_INTERVAL_MINUTES > 0:
            _schedule_periodic_check(session_id)

        return jsonify({"success": True, "session_id": session_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<int:session_id>/end', methods=['POST'])
def end_session(session_id):
    """End an attendance session."""
    try:
        # Cancel any pending periodic check
        if session_id in active_timers:
            active_timers[session_id].cancel()
            del active_timers[session_id]

        service.end_session(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<int:session_id>', methods=['GET'])
def get_session(session_id):
    """Get session details."""
    sessions = service.list_sessions()
    session_data = next((s for s in sessions if s['id'] == session_id), None)
    if session_data:
        return jsonify(session_data)
    return jsonify({"error": "Session not found"}), 404


# ---------------------------------------------------------------------------
# API Routes - Attendance Processing
# ---------------------------------------------------------------------------

@app.route('/api/session/<int:session_id>/capture', methods=['POST'])
def process_capture(session_id):
    """Process a captured frame for attendance."""
    data = request.get_json()
    image_data = data.get('image')
    snapshot_number = data.get('snapshot_number', 0)
    is_periodic = data.get('is_periodic', False)

    if not image_data:
        return jsonify({"success": False, "error": "No image provided."}), 400

    frame = decode_image(image_data)
    if frame is None:
        return jsonify({"success": False, "error": "Could not decode image."}), 400

    try:
        results = service.process_frame(
            session_id, frame,
            snapshot_number=snapshot_number,
            is_periodic=is_periodic
        )

        # Annotate frame for display
        annotated = annotate_frame(frame, results)
        annotated_b64 = encode_image(annotated)

        # Update dashboard data
        present = service.get_still_present(session_id)
        left = service.get_left_students(session_id)

        # Convert numpy arrays to Python lists for JSON serialization
        serializable_results = []
        for r in results:
            sr = dict(r)
            if "bbox" in sr and hasattr(sr["bbox"], "tolist"):
                sr["bbox"] = sr["bbox"].tolist()
            serializable_results.append(sr)

        return jsonify({
            "success": True,
            "results": serializable_results,
            "annotated_image": annotated_b64,
            "present": present,
            "left": left,
            "caption": "Periodic check" if is_periodic else "Manual capture"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<int:session_id>/dashboard', methods=['GET'])
def get_dashboard(session_id):
    """Get current dashboard data (present/left students)."""
    try:
        present = service.get_still_present(session_id)
        left = service.get_left_students(session_id)
        return jsonify({
            "success": True,
            "present": present,
            "left": left
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<int:session_id>/should-check', methods=['GET'])
def should_check(session_id):
    """Check if periodic check is due."""
    # Get last check time from session storage or calculate
    last_check = session.get(f'last_periodic_check_{session_id}')
    if last_check:
        last_check = datetime.fromisoformat(last_check)

    should_run = service.should_run_periodic_check(session_id, last_check)
    return jsonify({"should_check": should_run})


# ---------------------------------------------------------------------------
# API Routes - Reports
# ---------------------------------------------------------------------------

@app.route('/api/session/<int:session_id>/attendance', methods=['GET'])
def get_session_attendance(session_id):
    """Get attendance records for a session."""
    try:
        records = service.get_session_attendance(session_id)
        return jsonify(records)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<int:session_id>/export', methods=['GET'])
def export_csv(session_id):
    """Export session attendance as CSV."""
    try:
        records = service.get_session_attendance(session_id)
        if not records:
            return jsonify({"success": False, "error": "No data to export."}), 400

        df = pd.DataFrame(records)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'session_{session_id}_attendance.csv'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Periodic Check Logic (Server-side)
# ---------------------------------------------------------------------------

def _schedule_periodic_check(session_id):
    """Schedule the next periodic check for a session."""
    if session_id in active_timers:
        active_timers[session_id].cancel()

    interval_seconds = config.PERIODIC_CHECK_INTERVAL_MINUTES * 60

    def check_wrapper():
        # This runs server-side; we notify the client via SocketIO or 
        # the client polls for it
        socketio.emit('periodic_check_due', {'session_id': session_id}, namespace='/')
        # Reschedule
        _schedule_periodic_check(session_id)

    timer = threading.Timer(interval_seconds, check_wrapper)
    timer.daemon = True
    timer.start()
    active_timers[session_id] = timer


# ---------------------------------------------------------------------------
# SocketIO Events (for real-time periodic check notifications)
# ---------------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to attendance server'})


@socketio.on('join_session')
def handle_join_session(data):
    session_id = data.get('session_id')
    # Client joins a room for this session to receive periodic check notifications
    from flask_socketio import join_room
    join_room(f'session_{session_id}')
    emit('joined', {'session_id': session_id})


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Use 0.0.0.0 to accept connections from outside
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)
