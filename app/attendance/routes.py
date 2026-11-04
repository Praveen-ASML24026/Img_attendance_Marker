import os
import tempfile
from datetime import datetime, date
from typing import Set, Dict, List
from werkzeug.utils import secure_filename
from flask import render_template, redirect, url_for, request, flash, current_app, session as flask_session
from flask_login import login_required, current_user

from app.attendance import attendance
from app.models import Student, FaceEmbedding, AttendanceSession, AttendanceRecord
from app.extensions import db
from app.face_engine import FaceEngine, ClassroomProcessor
from app.google_sheets import GoogleSheetsSync

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}


def is_allowed_image(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


@attendance.route("/upload", methods=["GET", "POST"])
@login_required
def upload_classroom():
    """
    Phase 4: Classroom Photo Upload & Multi-Face Recognition.
    Accepts 1 to 3 classroom photographs, applies small-face tiling,
    runs ArcFace matching against enrolled students, and builds review session.
    """
    if request.method == "POST":
        session_date_str = request.form.get("session_date", "").strip()
        try:
            session_date = datetime.strptime(session_date_str, "%Y-%m-%d").date() if session_date_str else date.today()
        except ValueError:
            session_date = date.today()

        threshold_val = float(request.form.get("threshold", 0.50))

        # Check encryption key
        encryption_key = current_app.config.get("ENCRYPTION_KEY") or os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            flash("System encryption key is not configured.", "error")
            return redirect(url_for("attendance.upload_classroom"))

        # Retrieve classroom photo files
        uploaded_photos = []
        for key in ["photo_1", "photo_2", "photo_3"]:
            f = request.files.get(key)
            if f and f.filename and is_allowed_image(f.filename):
                uploaded_photos.append(f)

        if not uploaded_photos:
            flash("Please upload at least one classroom photograph.", "error")
            return redirect(url_for("attendance.upload_classroom"))

        # Query all enrolled students and decrypt their stored embeddings
        enrolled_students = Student.query.filter_by(is_enrolled=True).all()
        if not enrolled_students:
            flash("No students have been biometrically enrolled yet. Please enroll students first.", "error")
            return redirect(url_for("students.list_students"))

        engine = FaceEngine(recognition_threshold=threshold_val)
        processor = ClassroomProcessor(engine=engine, recognition_threshold=threshold_val)

        enrolled_db: Dict[str, List] = {}
        student_metadata: Dict[str, str] = {}

        for student in enrolled_students:
            embeddings_list = []
            for emb_rec in student.embeddings:
                try:
                    decrypted_vec = engine.decrypt_embedding(emb_rec.encrypted_embedding, encryption_key)
                    embeddings_list.append(decrypted_vec)
                except Exception:
                    pass

            if embeddings_list:
                enrolled_db[student.student_id] = embeddings_list
                student_metadata[student.student_id] = student.name

        # Process classroom photos inside temporary sandbox
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_paths = []
            for idx, photo_file in enumerate(uploaded_photos):
                safe_name = f"classroom_{idx + 1}_{secure_filename(photo_file.filename)}"
                path = os.path.join(temp_dir, safe_name)
                photo_file.save(path)
                temp_paths.append(path)

            results = processor.process_session_photos(
                photo_inputs=temp_paths,
                enrolled_db=enrolled_db,
                student_metadata=student_metadata,
                threshold=threshold_val
            )

        # Temporary files are deleted by the OS at this point

        # Create AttendanceSession in database
        att_session = AttendanceSession(
            session_date=session_date,
            total_detected=results["total_detections"],
            recognized_count=results["recognized_count"],
            unknown_count=results["unknown_count"],
            is_submitted=False,
            sync_status="Pending Submission",
            teacher_id=current_user.id
        )
        db.session.add(att_session)
        db.session.flush()

        # Build AttendanceRecord for every registered student
        all_students = Student.query.order_by(Student.name).all()
        recognized_map = results["recognized_students"]

        for stu in all_students:
            if stu.student_id in recognized_map:
                match_data = recognized_map[stu.student_id]
                rec = AttendanceRecord(
                    session_id=att_session.id,
                    student_id=stu.id,
                    status="P",
                    confidence=match_data["confidence"],
                    is_manually_corrected=False
                )
            else:
                rec = AttendanceRecord(
                    session_id=att_session.id,
                    student_id=stu.id,
                    status="A",
                    confidence=None,
                    is_manually_corrected=False
                )
            db.session.add(rec)

        db.session.commit()

        # Store unknown face crops temporarily in flask_session for the teacher review screen
        flask_session[f"unknown_faces_{att_session.id}"] = results["unknown_faces"]

        flash(
            f"Classroom recognition complete! {results['recognized_count']} student(s) recognized, {results['unknown_count']} unknown face(s) detected. Please review and confirm attendance below.",
            "success"
        )
        return redirect(url_for("attendance.review_session", session_id=att_session.id))

    return render_template(
        "attendance_upload.html",
        today_date=date.today().strftime("%Y-%m-%d"),
        enrolled_count=Student.query.filter_by(is_enrolled=True).count(),
        total_students=Student.query.count()
    )


@attendance.route("/review/<int:session_id>", methods=["GET", "POST"])
@login_required
def review_session(session_id):
    """
    Phase 5 & Phase 7: Interactive Teacher Attendance Review & Google Sheets Sync.
    Allows teachers to inspect recognized students, confirm or correct attendance,
    view unknown face crops, and trigger final Google Sheets synchronization.
    """
    att_session = db.get_or_404(AttendanceSession, session_id)

    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_status":
            # Individual student status toggle
            record_id = request.form.get("record_id", type=int)
            new_status = request.form.get("status")
            if record_id and new_status in ["P", "A"]:
                rec = db.session.get(AttendanceRecord, record_id)
                if rec and rec.session_id == att_session.id:
                    rec.status = new_status
                    rec.is_manually_corrected = True
                    rec.reviewed_at = datetime.now()
                    db.session.commit()
                    flash(f"Updated status for {rec.student.name} to {'Present' if new_status == 'P' else 'Absent'}.", "success")

            return redirect(url_for("attendance.review_session", session_id=att_session.id))

        elif action == "confirm_session":
            # 1. Final confirmation of attendance
            att_session.is_submitted = True
            present_count = AttendanceRecord.query.filter_by(session_id=att_session.id, status="P").count()
            att_session.recognized_count = present_count

            # 2. Synchronize with Google Sheets if configured (Phase 7)
            sheets_sync = GoogleSheetsSync()
            sync_msg = ""
            if sheets_sync.is_configured():
                records = AttendanceRecord.query.filter_by(session_id=att_session.id).join(Student).all()
                att_map = {r.student.student_id: r.status for r in records}
                name_map = {r.student.student_id: r.student.name for r in records}

                success, message = sheets_sync.sync_attendance(
                    session_date=att_session.session_date,
                    attendance_map=att_map,
                    student_names=name_map
                )
                if success:
                    att_session.synced_at = datetime.now()
                    att_session.sync_status = "Synced"
                    sync_msg = f" & Google Sheets updated successfully ({present_count} Present)."
                else:
                    att_session.sync_status = "Sync Failed"
                    sync_msg = f" (Google Sheets notice: {message})"
            else:
                att_session.sync_status = "Saved Locally (Sheets unconfigured)"
                sync_msg = " (Saved in local database. Set Google Service Account credentials to enable Sheets sync)."

            db.session.commit()

            # Clean up unknown faces from flask_session
            flask_session.pop(f"unknown_faces_{att_session.id}", None)

            flash(f"Attendance for {att_session.session_date.strftime('%d-%m-%Y')} confirmed!{sync_msg}", "success")
            return redirect(url_for("attendance.attendance_history"))

    records = AttendanceRecord.query.filter_by(session_id=att_session.id).join(Student).order_by(Student.name).all()
    unknown_faces = flask_session.get(f"unknown_faces_{att_session.id}", [])

    present_count = sum(1 for r in records if r.status == "P")
    absent_count = sum(1 for r in records if r.status == "A")

    sheets_configured = GoogleSheetsSync().is_configured()

    return render_template(
        "attendance_review.html",
        session=att_session,
        records=records,
        unknown_faces=unknown_faces,
        present_count=present_count,
        absent_count=absent_count,
        sheets_configured=sheets_configured
    )


@attendance.route("/sync/<int:session_id>", methods=["POST"])
@login_required
def sync_sheets(session_id):
    """Manually triggers Google Sheets synchronization for a confirmed session."""
    att_session = db.get_or_404(AttendanceSession, session_id)

    sheets_sync = GoogleSheetsSync()
    if not sheets_sync.is_configured():
        flash("Google Sheets credentials or Sheet ID are not configured in .env.", "error")
        return redirect(url_for("attendance.review_session", session_id=att_session.id))

    records = AttendanceRecord.query.filter_by(session_id=att_session.id).join(Student).all()
    att_map = {r.student.student_id: r.status for r in records}
    name_map = {r.student.student_id: r.student.name for r in records}

    success, message = sheets_sync.sync_attendance(
        session_date=att_session.session_date,
        attendance_map=att_map,
        student_names=name_map
    )

    if success:
        att_session.synced_at = datetime.now()
        att_session.sync_status = "Synced"
        db.session.commit()
        flash(message, "success")
    else:
        att_session.sync_status = "Sync Failed"
        db.session.commit()
        flash(message, "error")

    return redirect(url_for("attendance.review_session", session_id=att_session.id))


@attendance.route("/history")
@login_required
def attendance_history():
    """Lists all historical attendance sessions."""
    sessions = AttendanceSession.query.order_by(AttendanceSession.session_date.desc(), AttendanceSession.created_at.desc()).all()
    return render_template("attendance_history.html", sessions=sessions)
