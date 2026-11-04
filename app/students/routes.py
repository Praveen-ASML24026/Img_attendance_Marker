import os
import tempfile
from typing import Set
from werkzeug.utils import secure_filename
from flask import render_template, redirect, url_for, request, flash, current_app
from flask_login import login_required

from app.students import students
from app.models import Student, FaceEmbedding
from app.extensions import db
from app.face_engine import FaceEngine

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".webm"}


def is_allowed_file(filename: str, allowed_extensions: Set[str]) -> bool:
    """Verifies that the uploaded file has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


@students.route("/")
@login_required
def list_students():
    """Lists all registered students and their biometric enrollment status."""
    student_list = Student.query.order_by(Student.name).all()

    return render_template(
        "students.html",
        students=student_list
    )


@students.route("/enroll", methods=["GET", "POST"])
@login_required
def enroll():
    """Step 1: Register student details (Name and Student ID)."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        student_id = request.form.get("student_id", "").strip()

        if not name or not student_id:
            flash("Name and Student ID are required.", "error")
            return redirect(url_for("students.enroll"))

        existing = Student.query.filter_by(
            student_id=student_id
        ).first()

        if existing:
            flash("This Student ID already exists.", "error")
            return redirect(url_for("students.enroll"))

        student = Student(
            name=name,
            student_id=student_id,
            is_enrolled=False
        )

        db.session.add(student)
        db.session.commit()

        flash(
            f"Student '{student.name}' registered successfully! Proceed with biometric enrollment below.",
            "success"
        )

        return redirect(url_for("students.enroll_biometrics", student_id=student.id))

    return render_template("enroll.html")


@students.route("/biometrics/<int:student_id>", methods=["GET", "POST"])
@login_required
def enroll_biometrics(student_id):
    """
    Step 2: Biometric Enrollment & Re-enrollment Pipeline.
    Collects multiple face angles and short video, extracts 512-D ArcFace
    embeddings, encrypts them via Fernet, and immediately deletes temporary media.
    """
    student = db.get_or_404(Student, student_id)

    if request.method == "POST":
        # 1. Informed Consent Validation
        consent = request.form.get("consent")
        if not consent:
            flash("Informed biometric consent is mandatory before collecting or storing facial data.", "error")
            return redirect(url_for("students.enroll_biometrics", student_id=student.id))

        # 2. Cryptographic Key Check
        encryption_key = current_app.config.get("ENCRYPTION_KEY") or os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            flash("System encryption key is not configured. Biometrics cannot be securely stored.", "error")
            return redirect(url_for("students.enroll_biometrics", student_id=student.id))

        # 3. Retrieve and Validate Uploaded Files
        photo_keys = ["photo_front", "photo_left", "photo_right", "photo_up", "photo_down"]
        uploaded_photos = {}
        for key in photo_keys:
            f = request.files.get(key)
            if f and f.filename and is_allowed_file(f.filename, ALLOWED_IMAGE_EXTENSIONS):
                uploaded_photos[key] = f

        video_file = request.files.get("face_video")
        has_video = video_file and video_file.filename and is_allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS)

        if not uploaded_photos.get("photo_front") and not has_video:
            flash("Please upload at least the Front-Facing photograph or a short face video clip.", "error")
            return redirect(url_for("students.enroll_biometrics", student_id=student.id))

        engine = FaceEngine()
        extracted_embeddings = []
        angle_logs = []

        # 4. Isolated Temporary Directory (Automatically deleted on exit)
        with tempfile.TemporaryDirectory() as temp_dir:

            # A. Process Angle Photos
            for angle_key, file_obj in uploaded_photos.items():
                angle_name = angle_key.replace("photo_", "")
                safe_name = f"{angle_name}_{secure_filename(file_obj.filename)}"
                temp_path = os.path.join(temp_dir, safe_name)
                file_obj.save(temp_path)

                faces = engine.detect_faces(temp_path)
                if len(faces) == 0:
                    angle_logs.append(f"{angle_name.capitalize()}: No face detected")
                elif len(faces) > 1:
                    angle_logs.append(f"{angle_name.capitalize()}: Multiple faces detected (skipped)")
                else:
                    face = faces[0]
                    if face["det_score"] >= engine.det_thresh:
                        extracted_embeddings.append(face["normed_embedding"])
                        angle_logs.append(f"{angle_name.capitalize()} (pose={face['pose_label']})")
                    else:
                        angle_logs.append(f"{angle_name.capitalize()}: Low confidence ({face['det_score']:.2f})")

            # B. Process Video Clip
            if has_video:
                safe_video_name = f"video_{secure_filename(video_file.filename)}"
                temp_video_path = os.path.join(temp_dir, safe_video_name)
                video_file.save(temp_video_path)

                video_faces = engine.extract_from_video(temp_video_path, max_samples=6)
                if video_faces:
                    for vf in video_faces:
                        extracted_embeddings.append(vf["normed_embedding"])
                    angle_logs.append(f"Video: {len(video_faces)} keyframes")
                else:
                    angle_logs.append("Video: No clear frames detected")

        # At this point, temp_dir and all raw photos/videos are permanently deleted by the OS

        # 5. Check if any valid embeddings were extracted
        if not extracted_embeddings:
            flash("No high-quality facial biometric samples could be extracted. Please ensure proper lighting and clear visibility.", "error")
            return redirect(url_for("students.enroll_biometrics", student_id=student.id))

        # 6. Encrypt and Persist Biometric Records
        try:
            # Re-enrollment: delete prior embeddings for this student
            if student.is_enrolled:
                FaceEmbedding.query.filter_by(student_id=student.id).delete()

            # Insert freshly encrypted embeddings
            for emb in extracted_embeddings:
                encrypted_blob = engine.encrypt_embedding(emb, encryption_key)
                embedding_record = FaceEmbedding(
                    student_id=student.id,
                    encrypted_embedding=encrypted_blob
                )
                db.session.add(embedding_record)

            student.is_enrolled = True
            db.session.commit()

            summary = ", ".join(angle_logs)
            flash(
                f"Biometrics enrolled successfully for {student.name}! ({len(extracted_embeddings)} embeddings secured). Details: {summary}",
                "success"
            )
            return redirect(url_for("students.list_students"))

        except Exception as e:
            db.session.rollback()
            flash(f"Database error while saving encrypted biometrics: {str(e)}", "error")
            return redirect(url_for("students.enroll_biometrics", student_id=student.id))

    return render_template(
        "enroll_biometrics.html",
        student=student
    )


@students.route("/edit/<int:student_id>", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    """Edits student name or ID while preserving FaceEmbedding records."""
    student = db.get_or_404(Student, student_id)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        new_student_id = request.form.get("student_id", "").strip()

        if not name or not new_student_id:
            flash("Name and Student ID are required.", "error")
            return redirect(
                url_for("students.edit_student", student_id=student.id)
            )

        existing = Student.query.filter(
            Student.student_id == new_student_id,
            Student.id != student.id
        ).first()

        if existing:
            flash("This Student ID already exists.", "error")
            return redirect(
                url_for("students.edit_student", student_id=student.id)
            )

        student.name = name
        student.student_id = new_student_id

        try:
            db.session.commit()
            flash("Student details updated successfully.", "success")

            return redirect(url_for("students.list_students"))

        except Exception:
            db.session.rollback()
            flash("Unable to update student details.", "error")

    return render_template(
        "edit_student.html",
        student=student
    )
