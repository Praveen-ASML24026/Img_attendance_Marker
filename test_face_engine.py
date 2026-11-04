"""
Phase 2 Test Suite: Standalone Face Recognition Engine
Tests:
1. InsightFace + ArcFace model loading and initialization.
2. Video frame extraction and single/multi-face detection on real video (prav-face.mp4).
3. 512-dimensional embedding dimension, normalization, and quality metrics.
4. Cosine similarity calculation.
5. Known student vs. Unknown face identification logic.
6. Fernet cryptographic encryption and lossless decryption round-trip.
"""

import sys
import os
import numpy as np
from cryptography.fernet import Fernet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.face_engine.engine import FaceEngine

def run_tests():
    print("=" * 70)
    print("PHASE 2: STANDALONE FACE RECOGNITION ENGINE TEST SUITE")
    print("=" * 70)

    # 1. Initialize Engine
    print("\n[Step 1/5] Initializing Face Engine (buffalo_l on CPU)...")
    engine = FaceEngine(recognition_threshold=0.50)
    assert engine.analyzer is not None, "InsightFace analyzer failed to initialize"
    print("  [PASS] FaceEngine initialized successfully with SCRFD & ArcFace models")

    # 2. Test Video Processing on prav-face.mp4
    video_path = os.path.join(BASE_DIR, "prav-face.mp4")
    print(f"\n[Step 2/5] Testing Face Detection & Extraction on: {os.path.basename(video_path)}...")
    assert os.path.isfile(video_path), f"Test video not found at: {video_path}"

    sampled_faces = engine.extract_from_video(video_path, max_samples=5, frame_stride=10)
    assert len(sampled_faces) > 0, "No faces detected across video frames"
    print(f"  [PASS] Successfully extracted {len(sampled_faces)} high-quality face samples from video")

    sample = sampled_faces[0]
    print(f"  [PASS] Sample details:")
    print(f"         - Bounding Box: {sample['bbox']} (Width: {sample['width']}px, Height: {sample['height']}px)")
    print(f"         - Detection Confidence: {sample['det_score']:.4f}")
    print(f"         - Sharpness (Laplacian Var): {sample['sharpness']:.2f}")
    print(f"         - Head Pose: {sample['pose_label']} (pitch={sample['pose'][0]:.1f}, yaw={sample['pose'][1]:.1f}, roll={sample['pose'][2]:.1f})")

    # 3. Verify ArcFace Embedding Dimensions & Normalization
    print("\n[Step 3/5] Verifying 512-D ArcFace Embedding Attributes...")
    embedding = sample["embedding"]
    normed_embedding = sample["normed_embedding"]

    assert embedding.shape == (512,), f"Expected 512-D vector, got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected float32 dtype, got {embedding.dtype}"
    norm = np.linalg.norm(normed_embedding)
    assert abs(norm - 1.0) < 1e-4, f"Normalized embedding norm is not 1.0: {norm}"
    print(f"  [PASS] Embedding dimension: {embedding.shape[0]} float32 values")
    print(f"  [PASS] Unit L2 normalization verified: norm = {norm:.4f}")

    # 4. Test Cosine Similarity & Unknown Identification Logic
    print("\n[Step 4/5] Testing Cosine Similarity & Recognition Logic...")
    # Similarity with self must be 1.0
    self_sim = engine.cosine_similarity(normed_embedding, normed_embedding)
    assert abs(self_sim - 1.0) < 1e-4, f"Self-similarity should be 1.0, got {self_sim}"
    print(f"  [PASS] Self-similarity check: {self_sim:.4f}")

    # Similarity between two frames of the same student in video
    if len(sampled_faces) > 1:
        other_sample = sampled_faces[1]
        intra_sim = engine.cosine_similarity(sample["normed_embedding"], other_sample["normed_embedding"])
        print(f"  [PASS] Intra-person similarity (same student, different frames): {intra_sim:.4f}")
        assert intra_sim >= 0.50, f"Expected intra-person similarity >= 0.50, got {intra_sim}"

    # Build enrolled DB with student 'Praveen'
    enrolled_db = {
        "ASML24026": [sample["normed_embedding"]]
    }

    # Query with another frame of the same student -> Match
    matched_id, score, is_match = engine.match_face(sampled_faces[-1]["normed_embedding"], enrolled_db, threshold=0.50)
    assert is_match is True, f"Failed to match enrolled student: score={score}"
    assert matched_id == "ASML24026", f"Matched wrong student: {matched_id}"
    print(f"  [PASS] Enrolled match test: Matched student '{matched_id}' with similarity {score:.4f} (Threshold: 0.50)")

    # Query with a synthetic/different vector -> Must be identified as Unknown
    random_face_vector = np.random.randn(512).astype(np.float32)
    random_face_vector /= np.linalg.norm(random_face_vector)

    unknown_id, unknown_score, is_unknown_match = engine.match_face(random_face_vector, enrolled_db, threshold=0.50)
    assert is_unknown_match is False, f"False positive match on unknown face! Score: {unknown_score}"
    assert unknown_id is None, f"Expected None for unknown face, got: {unknown_id}"
    print(f"  [PASS] Unknown face test: Correctly rejected as Unknown (Best score: {unknown_score:.4f} < Threshold 0.50)")

    # 5. Test Fernet Encryption & Lossless Decryption
    print("\n[Step 5/5] Testing Biometric Cryptography Pipeline...")
    key = Fernet.generate_key()
    encrypted_blob = engine.encrypt_embedding(normed_embedding, key)
    assert isinstance(encrypted_blob, bytes)
    assert len(encrypted_blob) > 2000, "Encrypted payload size too small"

    decrypted_embedding = engine.decrypt_embedding(encrypted_blob, key)
    assert np.array_equal(normed_embedding, decrypted_embedding), "Decrypted array does not match original!"
    lossless_sim = engine.cosine_similarity(normed_embedding, decrypted_embedding)
    assert abs(lossless_sim - 1.0) < 1e-6, "Lossless decryption failed!"
    print(f"  [PASS] Raw float32 array encrypted into {len(encrypted_blob)} bytes opaque ciphertext")
    print(f"  [PASS] Decrypted embedding reconstructed losslessly (Cosine Similarity: {lossless_sim:.6f})")

    print("\n" + "=" * 70)
    print("ALL PHASE 2 ENGINE TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

