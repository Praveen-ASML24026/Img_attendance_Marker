"""
End-to-End Test Suite: Classroom Attendance Workflow (Phases 4, 5, 6)
Tests:
1. Classroom photo multi-upload via Flask endpoint.
2. Tiling & multi-face detection in application context.
3. Automatic generation of AttendanceSession & AttendanceRecord rows.
4. Correct classification of Present (Recognized) vs Absent (Not Detected).
5. Teacher review interface interaction: manual status correction.
6. Final attendance confirmation and session locking.
7. Attendance history inspection.
"""

import sys
import os
import io
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import create_app
from app.extensions import db
from app.models import User, Student, FaceEmbedding, AttendanceSession, AttendanceRecord
from app.face_engine import FaceEngine

def run_tests():
    print("=" * 70)
    print("PHASES 4, 5, 6: CLASSROOM ATTENDANCE & REVIEW TEST SUITE")
    print("=" * 70)

    test_db_file = os.path.join(BASE_DIR, "instance", "test_attendance_workflow.db")
    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
        except Exception:
            pass

    test_config = {
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{test_db_file}",
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    }

    app = create_app(test_config=test_config)

    with app.app_context():
        # 1. Setup Test Teacher & Enrolled Student
        print("\n[Step 1/6] Seeding Teacher and Enrolled Students...")
        teacher = User(username="att_teacher")
        teacher.set_password("TeacherPassword123!")
        db.session.add(teacher)

        # Enrolled student
        student_present = Student(name="Praveen R S", student_id="ASML24026", is_enrolled=True)
        # Absent student
        student_absent = Student(name="Absent John", student_id="ASML24099", is_enrolled=True)
        db.session.add_all([student_present, student_absent])
        db.session.commit()

        # Seed face embedding for student_present from real prav-face.mp4
        video_path = os.path.join(BASE_DIR, "prav-face.mp4")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not load video frame"

        engine = FaceEngine()
        faces = engine.detect_faces(frame)
        assert len(faces) > 0
        normed_emb = faces[0]["normed_embedding"]

        key = app.config.get("ENCRYPTION_KEY")
        encrypted_blob = engine.encrypt_embedding(normed_emb, key)
        emb_rec = FaceEmbedding(student_id=student_present.id, encrypted_embedding=encrypted_blob)
        db.session.add(emb_rec)
        db.session.commit()
        print(f"  [PASS] Enrolled student '{student_present.name}' with encrypted ArcFace vector")

        client = app.test_client()
        client.post("/login", data={"username": "att_teacher", "password": "TeacherPassword123!"})

        # 2. Upload Classroom Photos
        print("\n[Step 2/6] Uploading Classroom Photos to /attendance/upload...")
        _, encoded_img = cv2.imencode(".jpg", frame)
        img_bytes = encoded_img.tobytes()

        payload = {
            "session_date": "2026-11-04",
            "threshold": "0.50",
            "photo_1": (io.BytesIO(img_bytes), "classroom_center.jpg"),
            "photo_2": (io.BytesIO(img_bytes), "classroom_left.jpg")
        }

        res_upload = client.post(
            "/attendance/upload",
            data=payload,
            content_type="multipart/form-data",
            follow_redirects=False
        )
        assert res_upload.status_code == 302
        location = res_upload.headers.get("Location")
        assert "/attendance/review/" in location
        session_id = int(location.split("/")[-1])
        print(f"  [PASS] Classroom photos scanned and processed. Created Session #{session_id}")

        # 3. Verify Database Session & Records
        print("\n[Step 3/6] Verifying AttendanceSession and AttendanceRecord rows...")
        att_session = db.session.get(AttendanceSession, session_id)
        assert att_session is not None
        assert att_session.is_submitted is False
        assert len(att_session.records) == 2, f"Expected 2 records, found {len(att_session.records)}"

        rec_present = next(r for r in att_session.records if r.student_id == student_present.id)
        rec_absent = next(r for r in att_session.records if r.student_id == student_absent.id)

        assert rec_present.status == "P", f"Expected Present for recognized student, got {rec_present.status}"
        assert rec_present.confidence is not None and rec_present.confidence >= 0.50
        assert rec_absent.status == "A", f"Expected Absent for missing student, got {rec_absent.status}"
        assert rec_absent.confidence is None
        print(f"  [PASS] Present classification verified: {student_present.name} (Confidence: {rec_present.confidence:.4f})")
        print(f"  [PASS] Absent classification verified: {student_absent.name} (Status: {rec_absent.status})")

        # 4. Teacher Review Screen
        print("\n[Step 4/6] Inspecting Teacher Review Screen /attendance/review/<id>...")
        res_review = client.get(f"/attendance/review/{session_id}")
        assert res_review.status_code == 200
        assert b"Attendance Review" in res_review.data
        assert b"Praveen R S" in res_review.data
        assert b"Absent John" in res_review.data
        print("  [PASS] Review interface rendered student roster and metrics")

        # 5. Teacher Manual Correction
        print("\n[Step 5/6] Testing Teacher Manual Correction (Toggling Status)...")
        # Teacher manually marks Absent John as Present
        res_toggle = client.post(
            f"/attendance/review/{session_id}",
            data={
                "action": "update_status",
                "record_id": rec_absent.id,
                "status": "P"
            },
            follow_redirects=True
        )
        assert res_toggle.status_code == 200
        db.session.refresh(rec_absent)
        assert rec_absent.status == "P"
        assert rec_absent.is_manually_corrected is True
        print(f"  [PASS] Teacher manual correction verified: {student_absent.name} updated to Present (Flag: is_manually_corrected=True)")

        # 6. Final Confirmation & Submission
        print("\n[Step 6/6] Confirming and Submitting Attendance Session...")
        res_confirm = client.post(
            f"/attendance/review/{session_id}",
            data={"action": "confirm_session"},
            follow_redirects=True
        )
        assert res_confirm.status_code == 200
        db.session.refresh(att_session)
        assert att_session.is_submitted is True
        assert att_session.recognized_count == 2  # Both now Present after manual correction
        print(f"  [PASS] Attendance session confirmed and locked (is_submitted=True, Present count=2)")

        # Check history
        res_history = client.get("/attendance/history")
        assert res_history.status_code == 200
        assert b"Confirmed" in res_history.data
        print("  [PASS] Session confirmed in Attendance History")

        # Cleanup
        db.session.remove()

    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
            print("  [PASS] Isolated test database cleaned up")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL ATTENDANCE WORKFLOW TESTS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

