# Facial Recognition Attendance System

A classroom attendance system that uses facial recognition to automatically mark students present. It runs entirely on CPU, with no cloud inference dependency, and includes anti spoofing checks so a printed photo or a phone screen held up to the camera will not be accepted as a real student.

**Live demo:** https://facial-recognization-attendance.onrender.com

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Models Used](#models-used)
4. [System Architecture](#system-architecture)
5. [Configuration](#configuration)
6. [Tech Stack](#tech-stack)
7. [User Guide](#user-guide)
   - [Tab 1: Student Enrollment](#tab-1-student-enrollment)
   - [Tab 2: Taking Attendance](#tab-2-taking-attendance)
   - [Tab 3: Session Reports](#tab-3-session-reports)
   - [Quick Reference](#quick-reference)

---

## Overview

The system captures webcam snapshots, detects student faces, checks that the face is real (not a photo or screen), and matches it against enrolled reference data, all in one pipeline. Every frame that is captured goes through four stages:

| Stage | Component | What It Does | Model / Tool |
|---|---|---|---|
| 1 | Face Detection | Locates every face in the frame and returns bounding boxes | SCRFD 500MF (InsightFace buffalo_s) |
| 2 | Liveness Check | Rejects printed photos or screen replays before recognition runs | MiniFASNet V1SE + V2 (ONNX) |
| 3 | Face Recognition | Converts the face region into a 512 dimensional vector and searches for a match | MobileFaceNet (InsightFace buffalo_s) |
| 4 | Attendance Record | Marks the student present in the database, with duplicate protection | FAISS index + SQLite |

## How It Works

### Enrollment (one time setup per student)

Before a student can be recognized, they need to be enrolled:

1. The teacher captures 3 to 5 webcam photos of the student (frontal view, slight left and right turns, different lighting).
2. The system runs face detection and embedding extraction on each photo and keeps the highest confidence face.
3. All valid embeddings are averaged into a single vector and re normalized to unit length. This vector represents the student.
4. The vector is stored in the FAISS index under the student's database ID, while their name, code, and class go into SQLite.

### Taking Attendance (per snapshot)

During a class session, the teacher presses a capture button, or the system auto captures on a timer. Each snapshot goes through this sequence:

1. **Face Detection:** SCRFD scans the frame and returns a bounding box plus confidence score for each face found.
2. **Proximity Check:** if a face fills more than 35 percent of the frame, it is rejected as too close, which helps stop someone holding a printed photo right up to the lens.
3. **Liveness Check:** MiniFASNet evaluates the face at two crop scales and returns a real face probability. Below 0.65, the face is rejected as a spoof.
4. **Recognition:** MobileFaceNet embeds the verified face into a 512 dimensional vector, and FAISS searches for the nearest match. A cosine similarity of 0.40 or higher counts as a match.
5. **Dedup Guard:** if the student is already marked present for this session, the event is logged but no duplicate record is written.
6. **Mark Present:** an attendance record is written to SQLite with the confidence score and timestamp.

### Periodic Re-Verification

To stop a "mark and leave" workaround, the system automatically re-checks attendance every minute by default. If a student who was marked present no longer appears in a new snapshot, they get flagged as having left. A 7 minute grace period prevents a brief moment of looking away or being blocked from view from immediately counting as absent.

## Models Used

### SCRFD 500MF, Face Detector

A lightweight, single pass face detector trained with a distillation based approach to keep the parameter count under 500K while staying accurate across scales.

- Source: InsightFace buffalo_s model pack
- File size: about 2.5 MB
- Input: 320 x 320 px BGR image (configurable)
- Output: bounding box plus detection confidence per face
- Runtime: ONNXRuntime, CPUExecutionProvider

A single forward pass detects every face in the frame regardless of size, multi scale anchors handle students sitting at different distances, and the smaller 320x320 input keeps inference fast for a small classroom group.

### MobileFaceNet, Face Recognizer

A compact convolutional network trained with the ArcFace angular margin loss. It produces a 512 dimensional, L2 normalized embedding per face. Faces from the same person cluster closely together in this space, while faces from different people are separated by a clear angular margin.

- Source: InsightFace buffalo_s model pack
- File size: about 4 MB
- Output: 512 dimensional float32 vector, L2 normalized
- Training loss: ArcFace
- Similarity metric: cosine similarity (an inner product, since vectors are normalized)
- Match threshold: 0.40 or higher, configurable in `config.py`

Because the embeddings are already L2 normalized, cosine similarity simplifies to a plain inner product, which FAISS computes through `IndexFlatIP`, keeping the search both exact and fast.

### MiniFASNet V1SE + V2, Liveness / Anti Spoof

Detects presentation attacks, meaning someone holding up a printed photo or a phone or tablet screen instead of appearing in person. It looks at texture and frequency domain detail that real faces have and flat surface spoofs do not.

- Original source: minivision-ai/Silent-Face-Anti-Spoofing (Apache 2.0)
- Weights format: ONNX, converted from PyTorch .pth checkpoints
- Input: 80 x 80 px BGR crop (no /255 normalization, this detail matters)
- Output: 3 class logits, spoof type A, real, spoof type B
- Ensemble: V2 at 2.7x crop scale, plus V1SE at 4.0x crop scale
- Liveness threshold: 0.65 averaged real class probability across both models, configurable

The two models look at different crop windows around the detected face, a tighter 2.7x window and a wider 4.0x window. Together they catch both fine skin texture and wider context, like paper edges or screen bezels. Their outputs are averaged before the threshold is applied.

Note: the model receives the full captured frame plus the bounding box, not a pre cropped face. That surrounding context is what lets it spot a phone bezel or the edge of a printed page.

## System Architecture

### InsightFace buffalo_s Pack (Detector + Recognizer)

The SCRFD detector and MobileFaceNet recognizer come bundled together in InsightFace's buffalo_s model pack, managed as a single unit by the InsightFace library rather than as separate files.

- **Automatic download:** on first run, InsightFace downloads the buffalo_s pack (about 13 MB) and saves it to `~/.insightface/` on disk.
- **Initialization:** `FaceEngine` initializes `FaceAnalysis(name='buffalo_s')` and calls `prepare()` with the detection resolution (default 320 x 320), loading both ONNX models into ONNXRuntime sessions under CPUExecutionProvider.
- **Inference:** a single call to `app.get(image_bgr)` runs detection and embedding in one forward pass per model, no separate call needed per stage.
- **After first run:** everything works fully offline, no network access is required once the initial download is done.

### MiniFASNet ONNX Weights

Unlike the InsightFace models, the MiniFASNet weights are committed directly into the repository under `resources/anti_spoof_models/`, so nothing needs to be downloaded at runtime. They were converted once from the original PyTorch `.pth` checkpoints using `scripts/convert_anti_spoof_to_onnx.py`.

| File | Role |
|---|---|
| `2.7_80x80_MiniFASNetV2.onnx` | Tighter crop (2.7x face box), V2 architecture |
| `4_0_0_80x80_MiniFASNetV1SE.onnx` | Wider crop (4.0x face box), V1SE architecture (Squeeze and Excitation) |

At runtime, `liveness.py` reads the `config.ANTI_SPOOF_MODELS` list, opens an ONNXRuntime `InferenceSession` for each file, and caches the sessions in a module level variable. The input tensor name `input` and the output shape `(1, 3)` are fixed by the ONNX export.

### FAISS Vector Search

Student embeddings live in a FAISS `IndexFlatIP` (Flat Inner Product) index, wrapped with `IndexIDMap` so internal FAISS integer IDs map back to student IDs in SQLite.

- **Index type:** `IndexFlatIP` does an exact, exhaustive search, no approximation involved. For up to about 100 students this is effectively instant on CPU.
- **Similarity:** since embeddings are L2 normalized, inner product equals cosine similarity. A score of 1.0 is a perfect match, anything below 0.40 means no reliable match.
- **Persistence:** the index is saved to `data/face_embeddings.index` after every enrollment and loaded again at startup.
- **Update:** replacing a student's embedding means removing the old entry by ID first, then inserting the new one, since FAISS has no in place update.

### Database Schema (SQLite)

Three tables cover all application state. The schema works with both SQLite and PostgreSQL through SQLAlchemy, so switching databases only needs a change to `DB_URL` in `config.py`.

| Table | Key Columns | Purpose |
|---|---|---|
| `students` | id, student_code, name, class_id, last_avg_confidence, needs_reenrollment | Master list of enrolled students. ID also doubles as the FAISS vector ID. |
| `class_sessions` | id, classroom_id, subject, faculty, started_at, status | Tracks each attendance session (active or completed). |
| `attendance_records` | session_id, student_id, status, confidence, marked_at, is_periodic | One row per face event: present, left, unrecognized, spoof rejected, too close, already marked. |

## Configuration

All tunable values live in `src/config.py`. The defaults below are set up for a 5 to 10 student classroom with a single webcam.

| Parameter | Default | What to Change |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.40 | Raise to 0.45 to 0.50 if two similar looking students are getting confused for one another. |
| `LIVENESS_THRESHOLD` | 0.65 | Lower slightly, to around 0.60, if real students are being rejected because of poor lighting. |
| `MAX_FACE_FILL_RATIO` | 0.35 | The maximum share of the frame a single face can occupy before being rejected as too close. |
| `DETECTION_SIZE` | (320, 320) | Increase to 480x480 if the camera sits further from the students. |
| `PERIODIC_CHECK_INTERVAL_MINUTES` | 1 | Set to 0 to turn off automatic periodic checks. |
| `PRESENCE_GRACE_PERIOD_MINUTES` | 7 | How long a student can be missing from frame before being flagged as left. |
| `NUM_ENROLLMENT_IMAGES` | 5 | Minimum recommended enrollment photos per student. |

## Tech Stack

### Deployment

| Technology | Purpose |
|---|---|
| Render | Cloud hosting platform |
| Git | Version control |
| GitHub | Code repository |

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 | Page structure |
| JavaScript (vanilla) | All interactivity, no React or Vue |
| Bootstrap 5.3 | UI components, grid system, modals, tables |
| Bootstrap Icons | Icon set |
| Socket.IO Client | Receives real time periodic check notifications |
| Canvas API | Captures frames from the video stream |
| getUserMedia API | Accesses the webcam |
| Fetch API | Sends images to the backend over HTTP POST |

### Backend

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Flask | Web framework, routes, request handling, templating |
| Flask-SocketIO | Real time WebSocket communication for periodic check notifications |
| Gunicorn | Production WSGI server, with Eventlet for WebSocket support |
| InsightFace | Face detection (SCRFD 500MF) and recognition (MobileFaceNet) |
| ONNXRuntime | Inference engine for both face models and anti spoofing |
| FAISS-CPU | Local vector search for face embeddings |
| SQLAlchemy | ORM for the SQLite database |
| SQLite | Local database for students, sessions, and attendance records |
| OpenCV (headless) | Image processing, drawing bounding boxes |
| NumPy | Array operations for embeddings |
| Pandas | CSV export functionality |
| Pillow | Image format conversions |

---

## User Guide

This section walks through using the application itself, tab by tab, exactly as a teacher or admin would during a class.

### Tab 1: Student Enrollment

The Enrollment tab is where new students get registered. Each student needs their face captured from five different angles so recognition stays accurate later on.

**Filling Student Details**

1. Open the "Enroll Student" tab from the top navigation bar.
2. Make sure the camera is positioned and the student is ready for the photos.
3. Fill in the Student Details panel on the right:
   - Student ID / Code, a unique identifier, for example STU001
   - Full Name, the student's complete name
   - Class / Section, for example CS-101
4. Click "Start Camera" to turn on the webcam feed.

**Capturing Face Photos**

The system needs five photos of the student's face from different angles for the best recognition accuracy.

5. With the camera on, have the student look directly at it first.
6. Click "Capture" to take each photo.
7. Capture the photos in this order, watching the "Captured: X / 5" counter update each time:
   - Photo 1, straight on, looking directly at the camera
   - Photo 2, looking right
   - Photo 3, looking left
   - Photo 4, looking up
   - Photo 5, looking down
8. If any photo doesn't come out well, click "Reset All" and retake all five.
9. Once all 5 are captured, click the green "Enroll Student" button to save the student.
10. The student now shows up in the "Enrolled Students" table on the right.
11. Click "Stop" to turn off the camera once enrollment is finished.

**Managing Enrolled Students**

The Enrolled Students panel lists everyone currently registered, with their ID, Code, Name, and Class. A student can be removed if needed:

12. Find the student in the "Enrolled Students" table.
13. Click the delete icon next to their entry.
14. Confirm the deletion. This permanently removes the student along with their stored facial data.

### Tab 2: Taking Attendance

The Attendance tab runs live attendance sessions. The faculty member starts a session with the classroom details, and the system automatically checks attendance at regular intervals using face recognition.

**Starting a New Session**

15. Click "Take Attendance" in the navigation bar.
16. Fill in the session details under "Start New Session":
   - Classroom ID, for example ROOM-1
   - Subject, for example Mathematics
   - Faculty, for example Prof. Smith
17. Click "Start Session" to begin.

**Active Attendance Session**

Once the session is running, the screen switches to an active view with these parts:

- **Active Session Banner:** shows the session number, room ID, subject, and faculty name, with an "End Session" button.
- **Auto-Check Timer:** a green countdown bar showing when the next automatic capture will happen.
- **Camera Controls:** Start, Capture Snapshot, and Stop, for manual control of the camera.
- **Still Present Panel:** lists students currently detected and marked present.
- **Left the Room Panel:** tracks students who were present earlier but are no longer detected.

Session workflow:

18. Click "Start" to turn on the camera for attendance capture.
19. The system automatically checks attendance at the intervals shown on the countdown timer.
20. Students who are detected get marked "Present" automatically and appear in the "Still Present" panel.
21. Use "Capture Snapshot" any time for a manual check between the automatic ones.
22. Click "Stop" to turn off the camera without ending the session.
23. Click "End Session" (the red button) once class is over, to stop tracking and save the session data.

### Tab 3: Session Reports

The Reports tab is where you review attendance from past sessions, export it, and manage records that are no longer needed.

**Loading and Exporting Reports**

24. Click "Reports" in the navigation bar.
25. Open the dropdown under "Select Session" to see all available sessions.
26. Pick the session to review, shown with its room, subject, and date.
27. Click "Load Report" to pull up the attendance records.

The Attendance Records table shows, for each entry:

- **#**, the record's serial number
- **Student**, the student's name
- **Status**, present, absent, or left
- **Confidence**, the recognition confidence percentage
- **Marked At**, the timestamp when attendance was recorded
- **Snapshot**, the reference image captured at detection time
- **Auto**, whether the entry came from an automatic check or a manual one

28. To pull the data out, click "Export CSV". The file downloads with all attendance records for that session.
29. To see the session's metadata, like room, subject, faculty, and date, check the "Session Info" panel below the action buttons.

**Deleting Session Data**

To remove a session's attendance records:

30. Select the session from the dropdown.
31. Click "Delete Session" (outlined in red).
32. Confirm the deletion. This permanently removes all attendance records for that session.

### Quick Reference

| Tab | Action | How To |
|---|---|---|
| Enrollment | Add Student | Fill details, Start Camera, capture 5 angles, Enroll |
| Enrollment | Remove Student | Click the delete icon in the Enrolled Students sidebar |
| Attendance | Start Session | Fill Room / Subject / Faculty, click Start Session |
| Attendance | Auto Capture | Start the camera, the system checks automatically at intervals |
| Attendance | End Session | Click the End Session button (red) |
| Reports | View Records | Select a session from the dropdown, click Load Report |
| Reports | Export CSV | Select a session, click Export CSV |
| Reports | Delete Session | Select a session, click Delete Session |

---

Built by CodeWork.ai
