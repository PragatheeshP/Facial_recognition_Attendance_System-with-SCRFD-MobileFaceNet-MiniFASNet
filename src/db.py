"""
SQLite + SQLAlchemy storage for students, class sessions, and attendance
records. This mirrors the schema sketched in the original architecture doc
(students, classrooms/sessions, attendance_records) but trimmed down for a
single-classroom, 5-student deployment. Swap DB_URL in config.py for a
Postgres URL later with zero code changes if you outgrow SQLite.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from . import config

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)  # also used as the FAISS vector id
    student_code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    class_id = Column(String, nullable=True)
    enrolled_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_avg_confidence = Column(Float, nullable=True)
    needs_reenrollment = Column(Integer, default=0)  # 0/1 flag

    attendance_records = relationship("AttendanceRecord", back_populates="student")


class ClassSession(Base):
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(String, nullable=False)
    subject = Column(String, nullable=True)
    faculty = Column(String, nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="ACTIVE")  # ACTIVE | COMPLETED | CANCELLED

    attendance_records = relationship("AttendanceRecord", back_populates="session")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)  # null if unrecognized
    status = Column(String, nullable=False)  # PRESENT | LATE | UNRECOGNIZED | LEFT | TOO_CLOSE
    confidence = Column(Float, nullable=True)
    marked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # snapshot_number tracks which periodic check this record belongs to.
    # 0 = manual capture, 1+ = automatic periodic check.
    snapshot_number = Column(Integer, default=0)
    # True if this record was created by the automatic periodic check timer.
    is_periodic = Column(Integer, default=0)  # 0/1 flag

    session = relationship("ClassSession", back_populates="attendance_records")
    student = relationship("Student", back_populates="attendance_records")


_engine = create_engine(config.DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, future=True)


def init_db():
    Base.metadata.create_all(_engine)


def get_db():
    return SessionLocal()