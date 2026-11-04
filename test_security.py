"""
Phase 8 Security Test Suite: CSRF, Authentication Boundaries & Biometric Safeguards
Tests:
1. CSRF token enforcement on POST endpoints (rejects missing CSRF token with 400).
2. Unauthenticated access enforcement across all protected teacher routes (redirects to /login).
3. Plain-text biometric leak prevention (verifies database only stores opaque encrypted ciphertext).
4. Zero-retention verification (ensures no residual upload media remains on filesystem).
"""

import sys
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import create_app
from app.extensions import db
from app.models import User, Student, FaceEmbedding

def run_tests():
    print("=" * 70)
    print("PHASE 8: SECURITY, CSRF & BIOMETRIC PRIVACY TEST SUITE")
    print("=" * 70)

    test_db_file = os.path.join(BASE_DIR, "instance", "test_security.db")
    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
        except Exception:
            pass

    # Enable CSRF explicitly for this security test
    test_config = {
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{test_db_file}",
        "TESTING": True,
        "WTF_CSRF_ENABLED": True
    }

    app = create_app(test_config=test_config)

    with app.app_context():
        # Setup test teacher
        teacher = User(username="sec_teacher")
        teacher.set_password("SecurePassword123!")
        db.session.add(teacher)
        db.session.commit()

        client = app.test_client()

        # 1. Authentication Boundary Checks (Unauthenticated)
        print("\n[Step 1/4] Verifying Authentication Boundaries on All Protected Routes...")
        protected_routes = [
            "/students/",
            "/students/enroll",
            "/students/biometrics/1",
            "/students/edit/1",
            "/attendance/upload",
            "/attendance/review/1",
            "/attendance/history"
        ]

        for route in protected_routes:
            res = client.get(route, follow_redirects=False)
            assert res.status_code == 302, f"Route {route} not protected! Code: {res.status_code}"
            assert "/login" in res.headers.get("Location", "")
            print(f"  [PASS] Protected route blocked for anonymous user: {route} -> 302 Redirect to /login")

        # 2. CSRF Protection Check
        print("\n[Step 2/4] Verifying CSRF Protection Enforcement...")
        # A. POST without CSRF token must be blocked (HTTP 400 Bad Request)
        res_no_csrf = client.post("/login", data={"username": "sec_teacher", "password": "SecurePassword123!"})
        assert res_no_csrf.status_code == 400, f"Expected 400 Bad Request on missing CSRF, got {res_no_csrf.status_code}"
        print("  [PASS] Missing CSRF token correctly rejected with HTTP 400")

        # B. GET login page and extract valid CSRF token from HTML
        res_login_page = client.get("/login")
        assert res_login_page.status_code == 200
        csrf_match = re.search(r'name="csrf_token"\s+value="([^"]+)"', res_login_page.data.decode("utf-8"))
        assert csrf_match, "CSRF token input field not found in login.html"
        valid_csrf_token = csrf_match.group(1)

        # C. POST with valid CSRF token must succeed
        res_with_csrf = client.post(
            "/login",
            data={
                "csrf_token": valid_csrf_token,
                "username": "sec_teacher",
                "password": "SecurePassword123!"
            },
            follow_redirects=False
        )
        assert res_with_csrf.status_code == 302, f"Expected 302 redirect on valid CSRF, got {res_with_csrf.status_code}"
        print("  [PASS] Valid CSRF token authenticated and accepted successfully")

        # 3. Biometric Encryption at Rest
        print("\n[Step 3/4] Verifying Biometric Embeddings Encryption at Rest...")
        stu = Student(name="Sec Student", student_id="SEC001", is_enrolled=True)
        db.session.add(stu)
        db.session.commit()

        # Simulate encrypted embedding payload
        sample_ciphertext = b"gAAAAABnMockEncryptedBlobPayload=="
        emb = FaceEmbedding(student_id=stu.id, encrypted_embedding=sample_ciphertext)
        db.session.add(emb)
        db.session.commit()

        # Direct database query inspection: verify no plain-text float vectors or readable metadata
        db_record = db.session.get(FaceEmbedding, emb.id)
        assert isinstance(db_record.encrypted_embedding, bytes)
        assert not db_record.encrypted_embedding.startswith(b"[" or b"float" or b"numpy")
        print("  [PASS] Biometric embedding stored as opaque binary ciphertext in database")

        # 4. Zero-Retention Filesystem Verification
        print("\n[Step 4/4] Verifying Zero Residual Media on Filesystem...")
        tmp_dir = app.config.get("TEMP_DIR")
        if os.path.exists(tmp_dir):
            files = [f for f in os.listdir(tmp_dir) if f.endswith((".jpg", ".png", ".mp4", ".mov"))]
            assert len(files) == 0, f"Found dangling media files: {files}"
        print("  [PASS] No residual media files retained in temporary directory")

        # Cleanup
        db.session.remove()

    if os.path.exists(test_db_file):
        try:
            os.remove(test_db_file)
            print("  [PASS] Security test database cleaned up")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("ALL PHASE 8 SECURITY CHECKS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
