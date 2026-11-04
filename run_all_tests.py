"""
Master Test Runner for Face Recognition Attendance Management System
Executes the full test suite covering all modules:
1. Phase 1: Foundation (Auth, Registration, SQLite CRUD, Relationships)
2. Phase 2: Face Engine (SCRFD, ArcFace 512-D, Cosine Similarity, Quality & Pose)
3. Phase 3: Biometric Enrollment (Angle photos, Video sampling, Fernet encryption, Zero retention)
4. Phase 4: Classroom Recognition (Multi-face detection, Quadrant tiling, NMS deduplication)
5. Phase 5 & 6: Attendance Workflow (Teacher review interface, Status correction, History)
6. Phase 7: Google Sheets Synchronization (Auto-headers, Resubmission, Non-downgrade rule)
7. Phase 8: Security (CSRF protection, Protected route boundaries, Ciphertext verification)
"""

import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import test_foundation
import test_face_engine
import test_biometric_enrollment
import test_classroom_engine
import test_attendance_workflow
import test_google_sheets_sync
import test_security

SUITES = [
    ("Phase 1: Foundation & Auth", test_foundation.run_tests),
    ("Phase 2: Face Recognition Engine", test_face_engine.run_tests),
    ("Phase 3: Biometric Enrollment Pipeline", test_biometric_enrollment.run_tests),
    ("Phase 4: Classroom Tiling & Detection", test_classroom_engine.run_tests),
    ("Phase 5 & 6: Attendance Review & Session Models", test_attendance_workflow.run_tests),
    ("Phase 7: Google Sheets API Sync", test_google_sheets_sync.run_tests),
    ("Phase 8: Security Hardening & CSRF", test_security.run_tests),
]

def main():
    start_time = time.time()
    print("*" * 80)
    print("MASTER TEST SUITE EXECUTION — FACE ATTENDANCE MANAGEMENT SYSTEM")
    print("*" * 80)

    results = []

    for name, runner in SUITES:
        suite_start = time.time()
        print(f"\n>>> Running: {name} ...")
        try:
            runner()
            duration = time.time() - suite_start
            results.append((name, "PASSED", f"{duration:.2f}s"))
        except Exception as e:
            duration = time.time() - suite_start
            results.append((name, f"FAILED: {str(e)}", f"{duration:.2f}s"))
            print(f"FAILED in {name}: {e}")

    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print("FINAL TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Module / Phase Name':<45} | {'Result':<15} | {'Duration'}")
    print("-" * 80)

    all_passed = True
    for name, status, duration in results:
        status_text = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"{name:<45} | {status_text:<15} | {duration}")
        if status != "PASSED":
            all_passed = False

    print("=" * 80)
    if all_passed:
        print(f"OVERALL STATUS: ALL {len(results)} SUITES PASSED (100% SUCCESS) in {total_duration:.2f}s")
    else:
        print("OVERALL STATUS: ONE OR MORE SUITES FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()

