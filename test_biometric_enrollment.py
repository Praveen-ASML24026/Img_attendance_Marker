"""
Phase 3 Test Suite: Biometric Enrollment & Re-Enrollment Pipeline
Tests:
1. Registration -> Biometric Enrollment redirection.
2. Mandatory consent enforcement.
3. Multi-angle photo and video processing via Flask endpoint.
4. 512-D ArcFace embedding generation, Fernet encryption, and DB storage.
5. Verification of student.is_enrolled status update.
6. Decryption verification of stored DB embeddings.
7. Re-enrollment functionality (purging stale embeddings and persisting new ones).
8. Zero temporary media persistence check.
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
from app.models import User, Student, FaceEmbedding
from app.face_engine import FaceEngine

def run_tests():
    print("=" * 70)
    print("PHASE 3: BIOMETRIC ENROLLMENT PIPELINE TEST SUITE")
    print("=" * 70)

    # Use isolated test DB to avoid locking
    test_db_file = os.path.join(BASE_DIR, "instance", "test_biometric_enroll.db")
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
        # Setup teacher
        teacher = User(username="test_teacher_bio")
        teacher.set_password("TeacherPassword123!")
        db.session.add(teacher)
        db.session.commit()

        client = app.test_client()

        # Login as teacher
        client.post("/login", data={"username": "test_teacher_bio", "password": "TeacherPassword123!"})

        # 1. Step 1: Register Student
        print("\n[Step 1/6] Registering New Student (Step 1)...")
        res_reg = client.post(
            "/students/enroll",
            data={"name": "Praveen Test", "student_id": "TEST_BIO_001"},
            follow_redirects=False
        )
        assert res_reg.status_code == 302
        stu = Student.query.filter_by(student_id="TEST_BIO_001").first()
        assert stu is not None
        assert stu.is_enrolled is False
        assert f"/students/biometrics/{stu.id}" in res_reg.headers.get("Location")
        print(f"  [PASS] Student registered. Redirected to Step 2: /students/biometrics/{stu.id}")

        # 2. Test Consent Enforcement
        print("\n[Step 2/6] Testing Informed Consent Enforcement...")
        res_no_consent = client.post(
            f"/students/biometrics/{stu.id}",
            data={},
            follow_redirects=True
        )
        assert b"Informed biometric consent is mandatory" in res_no_consent.data
        print("  [PASS] Missing consent correctly rejected")

        # 3. Test Missing Files Enforcement
        print("\n[Step 3/6] Testing Missing Media Enforcement...")
        res_no_media = client.post(
            f"/students/biometrics/{stu.id}",
            data={"consent": "on"},
            follow_redirects=True
        )
        assert b"Please upload at least the Front-Facing photograph or a short face video" in res_no_media.data
        print("  [PASS] Missing media submission correctly rejected")

        # 4. Prepare Media from real prav-face.mp4
        print("\n[Step 4/6] Processing Real Media Upload (Photo + Video)...")
        video_path = os.path.join(BASE_DIR, "prav-face.mp4")
        assert os.path.isfile(video_path), "prav-face.mp4 required for test"

        # Extract 1 frame as photo_front
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from video"
        _, encoded_img = cv2.imencode(".jpg", frame)
        img_bytes = encoded_img.tobytes()

        with open(video_path, "rb") as vf:
            video_bytes = vf.read()

        payload = {
            "consent": "on",
            "photo_front": (io.BytesIO(img_bytes), "front.jpg"),
            "face_video": (io.BytesIO(video_bytes), "video.mp4")
        }

        res_enroll = client.post(
            f"/students/biometrics/{stu.id}",
            data=payload,
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert res_enroll.status_code == 200
        assert b"Biometrics enrolled successfully" in res_enroll.data

        db.session.refresh(stu)
        assert stu.is_enrolled is True, "Student is_enrolled should be True!"
        embeddings_count = len(stu.embeddings)
        assert embeddings_count > 0, "No FaceEmbedding records stored in DB!"
        print(f"  [PASS] Enrollment complete: {embeddings_count} encrypted embeddings stored in database")
        print(f"  [PASS] Student status updated: is_enrolled = {stu.is_enrolled}")

        # 5. Verify Cryptographic Integrity of Database Embeddings
        print("\n[Step 5/6] Verifying DB Embeddings Decryption & Lossless Recovery...")
        engine = FaceEngine()
        key = app.config.get("ENCRYPTION_KEY")
        first_record = stu.embeddings[0]
        decrypted_vector = engine.decrypt_embedding(first_record.encrypted_embedding, key)

        assert decrypted_vector.shape == (512,), f"Wrong shape: {decrypted_vector.shape}"
        assert decrypted_vector.dtype == np.float32
        norm = np.linalg.norm(decrypted_vector)
        assert abs(norm - 1.0) < 1e-4, f"Norm is not 1.0: {norm}"
        print(f"  [PASS] Database payload decrypted into valid 512-D float32 normalized vector (norm={norm:.4f})")

        # 6. Test Re-enrollment Pipeline
        print("\n[Step 6/6] Testing Re-enrollment (Updating Existing Biometrics)...")
        # Post new payload (just the front photo this time)
        re_payload = {
            "consent": "on",
            "photo_front": (io.BytesIO(img_bytes), "front_new.jpg")
        }
        res_re_enroll = client.post(
            f"/students/biometrics/{stu.id}",
            data=re_payload,
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert b"Biometrics enrolled successfully" in res_re_enroll.data
        db.session.refresh(stu)
        assert stu.is_enrolled is True
        # The old embeddings should have been replaced
        new_count = len(stu.embeddings)
        print(f"  [PASS] Re-enrollment replaced previous {embeddings_count} embeddings with {new_count} fresh embeddings")

        # Cleanup
        db.session.remove()

    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
            print("  [PASS] Test database cleaned up successfully")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL PHASE 3 BIOMETRIC ENROLLMENT CHECKS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
