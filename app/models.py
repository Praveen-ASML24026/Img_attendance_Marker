from datetime import datetime, timezone, date

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db, login_manager


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    is_enrolled = db.Column(db.Boolean, default=False, nullable=False)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    embeddings = db.relationship(
        "FaceEmbedding",
        backref="student",
        cascade="all, delete-orphan"
    )


class FaceEmbedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    student_id = db.Column(
        db.Integer,
        db.ForeignKey("student.id"),
        nullable=False
    )

    encrypted_embedding = db.Column(db.LargeBinary, nullable=False)

    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class AttendanceSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_date = db.Column(db.Date, nullable=False, default=date.today)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    total_detected = db.Column(db.Integer, default=0, nullable=False)
    recognized_count = db.Column(db.Integer, default=0, nullable=False)
    unknown_count = db.Column(db.Integer, default=0, nullable=False)
    is_submitted = db.Column(db.Boolean, default=False, nullable=False)

    synced_at = db.Column(db.DateTime(timezone=True), nullable=True)
    sync_status = db.Column(db.String(255), nullable=True)

    teacher_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    teacher = db.relationship("User", backref="attendance_sessions")

    records = db.relationship(
        "AttendanceRecord",
        backref="session",
        cascade="all, delete-orphan",
        order_by="AttendanceRecord.id"
    )


class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("attendance_session.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    status = db.Column(db.String(1), default="A", nullable=False)  # 'P' for Present, 'A' for Absent
    confidence = db.Column(db.Float, nullable=True)
    is_manually_corrected = db.Column(db.Boolean, default=False, nullable=False)
    reviewed_at = db.Column(db.DateTime(timezone=True), nullable=True)

    student = db.relationship("Student", backref="attendance_records")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
