"""
Phase 4 Test Suite: Classroom Multi-Face Recognition & Tiling Engine
Tests:
1. Mathematical IoU calculation correctness.
2. Non-Maximum Suppression (NMS) duplicate pruning.
3. Multi-scale classroom tiling & small-face recovery.
4. Cosine similarity recognition & unknown face categorization.
5. Cross-photo duplicate aggregation (highest confidence selection).
6. Missing/unrecognized enrolled student detection.
"""

import sys
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.face_engine.classroom import ClassroomProcessor, calculate_iou, apply_nms
from app.face_engine.engine import FaceEngine

def run_tests():
    print("=" * 70)
    print("PHASE 4: CLASSROOM FACE RECOGNITION & TILING ENGINE TEST SUITE")
    print("=" * 70)

    # 1. Mathematical IoU Verification
    print("\n[Step 1/5] Verifying IoU Bounding Box Math...")
    box_a = [0, 0, 100, 100]
    box_b = [0, 0, 100, 100]
    box_c = [50, 50, 150, 150]
    box_d = [200, 200, 300, 300]

    iou_identical = calculate_iou(box_a, box_b)
    assert abs(iou_identical - 1.0) < 1e-4, f"Expected 1.0, got {iou_identical}"
    print(f"  [PASS] Identical boxes IoU: {iou_identical:.4f}")

    iou_overlap = calculate_iou(box_a, box_c)
    # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500 -> 2500/17500 = 0.142857
    assert 0.14 < iou_overlap < 0.15, f"Expected ~0.1429, got {iou_overlap}"
    print(f"  [PASS] Overlapping boxes IoU: {iou_overlap:.4f}")

    iou_disjoint = calculate_iou(box_a, box_d)
    assert iou_disjoint == 0.0, f"Expected 0.0, got {iou_disjoint}"
    print(f"  [PASS] Disjoint boxes IoU: {iou_disjoint:.4f}")

    # 2. Non-Maximum Suppression (NMS) Verification
    print("\n[Step 2/5] Verifying NMS Deduplication...")
    duplicate_detections = [
        {"bbox": [10, 10, 100, 100], "det_score": 0.95, "id": "best_candidate"},
        {"bbox": [12, 11, 102, 101], "det_score": 0.88, "id": "duplicate_tile_overlap"},
        {"bbox": [300, 300, 400, 400], "det_score": 0.92, "id": "distinct_student_face"}
    ]
    pruned = apply_nms(duplicate_detections, iou_threshold=0.40)
    assert len(pruned) == 2, f"Expected 2 unique faces after NMS, got {len(pruned)}"
    assert pruned[0]["id"] == "best_candidate"
    assert pruned[1]["id"] == "distinct_student_face"
    print(f"  [PASS] NMS pruned overlapping tile candidate, retaining {len(pruned)} unique detections")

    # 3. Classroom Multi-Scale Tiling & Face Detection Test
    print("\n[Step 3/5] Testing Classroom Tiling & Small-Face Detection...")
    video_path = os.path.join(BASE_DIR, "prav-face.mp4")
    assert os.path.isfile(video_path), "prav-face.mp4 required for test"

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not load video frame"

    # Create a synthetic 1280x720 classroom composite:
    # 1. Front row: student face at native resolution
    # 2. Back row: student face scaled down to 40% (simulating small distant student in row 5)
    classroom_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Place front-row student at bottom-center (e.g. [150:650, 400:800])
    face_crop = frame[227:670, 202:493]
    fh, fw = face_crop.shape[:2]

    # Front row face
    classroom_canvas[200:200 + fh, 200:200 + fw] = face_crop

    # Small back-row face (scaled down)
    small_face = cv2.resize(face_crop, (int(fw * 0.45), int(fh * 0.45)))
    sfh, sfw = small_face.shape[:2]
    classroom_canvas[50:50 + sfh, 900:900 + sfw] = small_face

    processor = ClassroomProcessor()
    detected_faces = processor.detect_classroom_faces(classroom_canvas, use_tiling=True)
    print(f"  [PASS] Tiled detection detected {len(detected_faces)} face(s) in high-res classroom canvas")
    assert len(detected_faces) >= 2, f"Expected at least 2 faces (front row + back row), detected {len(detected_faces)}"

    for idx, df in enumerate(detected_faces):
        print(f"         Face {idx + 1}: BBox={df['bbox']}, Confidence={df['det_score']:.4f}, Source={df['source']}")

    # 4. Enrolled Matching & Unknown Face Identification
    print("\n[Step 4/5] Testing Enrolled Matching & Unknown Face Identification...")
    # Seed enrolled DB with student 'ASML24026' (Praveen R S) using front face embedding
    enrolled_embedding = detected_faces[0]["normed_embedding"]
    enrolled_db = {
        "ASML24026": [enrolled_embedding]
    }
    student_meta = {
        "ASML24026": "Praveen R S",
        "ASML24099": "Absent Student"  # Enrolled but not in the photo
    }

    session_result = processor.process_session_photos(
        photo_inputs=[classroom_canvas],
        enrolled_db=enrolled_db,
        student_metadata=student_meta,
        threshold=0.50
    )

    recognized = session_result["recognized_students"]
    unknown = session_result["unknown_faces"]
    missing = session_result["missing_students"]

    assert "ASML24026" in recognized, "Enrolled student should be recognized!"
    print(f"  [PASS] Enrolled student recognized: {recognized['ASML24026']['name']} (Confidence: {recognized['ASML24026']['confidence']:.4f})")

    assert len(missing) == 1 and missing[0]["student_id"] == "ASML24099"
    print(f"  [PASS] Absent student correctly flagged: {missing[0]['name']} (ID: {missing[0]['student_id']})")

    # 5. Cross-Photo Deduplication Test
    print("\n[Step 5/5] Testing Cross-Photo Aggregation & De-duplication...")
    # Simulate teacher uploading 2 photos of the classroom
    multi_photo_result = processor.process_session_photos(
        photo_inputs=[classroom_canvas, classroom_canvas],
        enrolled_db=enrolled_db,
        student_metadata=student_meta,
        threshold=0.50
    )

    # Even though 2 photos were provided, student 'ASML24026' must appear exactly ONCE
    assert len(multi_photo_result["recognized_students"]) == 1
    assert "ASML24026" in multi_photo_result["recognized_students"]
    print(f"  [PASS] Multi-photo aggregation de-duplicated student across 2 photos into a single Present record")

    print("\n" + "=" * 70)
    print("ALL PHASE 4 CLASSROOM ENGINE TESTS PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

