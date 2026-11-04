"""
Foundation Test Suite for Face Attendance Management System
Tests:
- Environment & Database configuration
- Teacher authentication workflow (login/logout/protected routes)
- Student registration & duplicate rejection
- Student editing & collision detection
- FaceEmbedding relationship preservation during edits
- Non-destructive test data cleanup
"""

import sys
import os
import tempfile

# Ensure face-attendance root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Use an isolated test database to prevent lock collisions with active DB Browser sessions
test_db_file = os.path.join(BASE_DIR, "instance", "test_foundation.db")
if os.path.exists(test_db_file):
    try:
        os.remove(test_db_file)
    except Exception:
        pass

os.environ["DATABASE_URL"] = f"sqlite:///{test_db_file}"

from app import create_app
from app.extensions import db
from app.models import User, Student, FaceEmbedding

def run_tests():
    print("=" * 70)
    print("PHASE 1: FOUNDATION VERIFICATION TEST SUITE (ISOLATED TEST DB)")
    print("=" * 70)

    test_config = {
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{test_db_file}",
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    }
    app = create_app(test_config=test_config)

    with app.app_context():
        # 1. Environment & Database Check
        print("\n[Step 1/5] Verifying Database & Schema...")
        tables = db.metadata.tables.keys()
        expected_tables = {"user", "student", "face_embedding"}
        assert expected_tables.issubset(tables), f"Missing tables! Found: {tables}"
        print("  [PASS] Schema tables verified: user, student, face_embedding")

        # Seed initial teacher and student matching instance/attendance.db
        baseline_teacher = User(username="praveen")
        baseline_teacher.set_password("TeacherPassword123!")
        baseline_student = Student(name="Praveen R S", student_id="ASML24026", is_enrolled=False)
        db.session.add_all([baseline_teacher, baseline_student])
        db.session.commit()

        existing_teachers = User.query.all()
        existing_students = Student.query.all()
        print(f"  [PASS] Seeded baseline teachers in test DB: {[t.username for t in existing_teachers]}")
        print(f"  [PASS] Seeded baseline students in test DB: {[(s.student_id, s.name) for s in existing_students]}")

        client = app.test_client()

        # 2. Authentication Flow
        print("\n[Step 2/5] Testing Authentication Flow...")
        res_unauth = client.get("/students/", follow_redirects=False)
        assert res_unauth.status_code == 302, f"Expected 302 redirect, got {res_unauth.status_code}"
        assert "/login" in res_unauth.headers.get("Location", ""), "Did not redirect to /login"
        print("  [PASS] Protected route /students/ redirects unauthenticated users to /login")

        res_bad_login = client.post(
            "/login",
            data={"username": "non_existent_teacher", "password": "WrongPassword123"},
            follow_redirects=True
        )
        assert b"Invalid username or password" in res_bad_login.data
        print("  [PASS] Invalid login credentials correctly rejected")

        # Create temporary test teacher if needed, or use existing
        test_teacher = User.query.filter_by(username="temp_test_teacher").first()
        if not test_teacher:
            test_teacher = User(username="temp_test_teacher")
            test_teacher.set_password("TempTestPass123!")
            db.session.add(test_teacher)
            db.session.commit()

        # Test Teacher Self-Registration (/register)
        res_mismatch = client.post(
            "/register",
            data={"username": "new_prof", "password": "Password123", "confirm_password": "MismatchPassword"},
            follow_redirects=True
        )
        assert b"Passwords do not match" in res_mismatch.data
        print("  [PASS] Password mismatch validation rejected in /register")

        res_reg_teacher = client.post(
            "/register",
            data={"username": "registered_prof", "password": "SecurePassword123!", "confirm_password": "SecurePassword123!"},
            follow_redirects=True
        )
        assert b"Account created successfully" in res_reg_teacher.data
        registered_user = User.query.filter_by(username="registered_prof").first()
        assert registered_user is not None
        assert registered_user.check_password("SecurePassword123!") is True
        print("  [PASS] Teacher self-registration (/register) created user with hashed credentials")

        res_dup_teacher = client.post(
            "/register",
            data={"username": "registered_prof", "password": "SecurePassword123!", "confirm_password": "SecurePassword123!"},
            follow_redirects=True
        )
        assert b"Username already exists" in res_dup_teacher.data
        print("  [PASS] Duplicate teacher registration prevented")

        res_good_login = client.post(
            "/login",
            data={"username": "registered_prof", "password": "SecurePassword123!"},
            follow_redirects=True
        )
        assert res_good_login.status_code == 200
        assert b"Student Management" in res_good_login.data
        print("  [PASS] Valid teacher login succeeded and redirected to Student Management")

        # 3. Student Registration Flow
        print("\n[Step 3/5] Testing Student Registration & Duplicate Prevention...")
        # Clean up any residual test records first
        Student.query.filter(Student.student_id.like("TEST_STU%")).delete()
        db.session.commit()

        res_reg = client.post(
            "/students/enroll",
            data={"name": "Test Student Alpha", "student_id": "TEST_STU001"},
            follow_redirects=True
        )
        assert b"registered successfully" in res_reg.data
        test_stu = Student.query.filter_by(student_id="TEST_STU001").first()
        assert test_stu is not None, "Test student was not saved to database"
        assert test_stu.is_enrolled is False, "New student should have is_enrolled=False"
        print("  [PASS] Student registration successful with pending biometric status")

        res_dup = client.post(
            "/students/enroll",
            data={"name": "Duplicate Student", "student_id": "TEST_STU001"},
            follow_redirects=True
        )
        assert b"This Student ID already exists" in res_dup.data
        print("  [PASS] Duplicate Student ID registration correctly rejected")

        # 4. Student Editing & Biometric Relationship Integrity
        print("\n[Step 4/5] Testing Student Editing & Biometric Integrity...")
        # Add a mock FaceEmbedding to test_stu to verify it survives edits
        mock_embedding = FaceEmbedding(
            student_id=test_stu.id,
            encrypted_embedding=b"encrypted_vector_mock_payload"
        )
        db.session.add(mock_embedding)
        db.session.commit()
        assert len(test_stu.embeddings) == 1, "FaceEmbedding was not attached"

        # Edit test student name and ID
        res_edit = client.post(
            f"/students/edit/{test_stu.id}",
            data={"name": "Test Student Alpha Modified", "student_id": "TEST_STU001_MOD"},
            follow_redirects=True
        )
        assert b"Student details updated successfully" in res_edit.data
        db.session.refresh(test_stu)
        assert test_stu.name == "Test Student Alpha Modified"
        assert test_stu.student_id == "TEST_STU001_MOD"
        assert len(test_stu.embeddings) == 1, "FaceEmbedding was lost during edit!"
        assert test_stu.embeddings[0].encrypted_embedding == b"encrypted_vector_mock_payload"
        print("  [PASS] Student details updated successfully while preserving FaceEmbedding records")

        # Register second test student to verify collision prevention on edit
        client.post(
            "/students/enroll",
            data={"name": "Test Student Beta", "student_id": "TEST_STU002"},
            follow_redirects=True
        )
        test_stu2 = Student.query.filter_by(student_id="TEST_STU002").first()
        assert test_stu2 is not None

        res_edit_conflict = client.post(
            f"/students/edit/{test_stu2.id}",
            data={"name": "Test Student Beta", "student_id": "TEST_STU001_MOD"},
            follow_redirects=True
        )
        assert b"This Student ID already exists" in res_edit_conflict.data
        print("  [PASS] Edit duplicate collision with another student's ID correctly blocked")

        # 5. Cleanup Test Records
        print("\n[Step 5/5] Cleaning up test artifacts...")
        db.session.remove()
        print("  [PASS] Test session closed.")

    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
            print("  [PASS] Isolated test database file removed.")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL FOUNDATION CHECKS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
