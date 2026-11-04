import os
import warnings
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from cryptography.fernet import Fernet
import insightface
from insightface.app import FaceAnalysis

# Suppress internal third-party scikit-image transform deprecation warnings from insightface
warnings.filterwarnings("ignore", category=FutureWarning)

class FaceEngine:
    """
    Core Face Recognition Engine utilizing InsightFace & ArcFace.
    Provides:
    - Face detection and landmark extraction (SCRFD)
    - 512-dimensional ArcFace feature embedding extraction
    - L2 normalization and cosine similarity matching
    - Image sharpness & pose angle classification
    - Video keyframe sampling and biometric extraction
    - Lossless Fernet encryption and decryption of float32 embeddings
    """

    _instance = None
    _analyzer = None

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.50,
        recognition_threshold: float = 0.50
    ):
        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.recognition_threshold = recognition_threshold
        self._ensure_analyzer_loaded()

    def _ensure_analyzer_loaded(self):
        """Lazily initialize and cache the InsightFace FaceAnalysis model."""
        if FaceEngine._analyzer is None:
            # Use local models directory if present for portability
            project_models_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
            local_models_path = os.path.join(project_models_dir, "models")
            root_dir = project_models_dir if os.path.isdir(local_models_path) else "~/.insightface"

            analyzer = FaceAnalysis(
                name=self.model_name,
                root=root_dir,
                providers=["CPUExecutionProvider"]
            )
            analyzer.prepare(
                ctx_id=0,
                det_size=self.det_size,
                det_thresh=self.det_thresh
            )
            FaceEngine._analyzer = analyzer
        self.analyzer = FaceEngine._analyzer

    @staticmethod
    def load_image(image_input: Any) -> Optional[np.ndarray]:
        """
        Loads an image from a file path, bytes, or returns it if already a numpy ndarray.
        Returns BGR image array or None if loading fails.
        """
        if isinstance(image_input, np.ndarray):
            return image_input
        elif isinstance(image_input, str):
            if not os.path.isfile(image_input):
                return None
            return cv2.imread(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return None

    def detect_faces(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detects all visible faces in an image and extracts their 512-D ArcFace embeddings.
        Returns a list of face dictionaries containing:
        - bbox: [x1, y1, x2, y2]
        - det_score: detection confidence float
        - landmarks: 5 keypoint coordinates (eyes, nose, mouth corners)
        - pose: [pitch, yaw, roll] in degrees
        - pose_label: 'front', 'left', 'right', 'up', 'down'
        - embedding: raw 512-D float32 vector
        - normed_embedding: unit L2-normalized 512-D float32 vector
        - sharpness: Laplacian variance score
        """
        img = self.load_image(image)
        if img is None:
            return []

        raw_faces = self.analyzer.get(img)
        processed_faces = []

        h, w = img.shape[:2]

        for face in raw_faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop face region to evaluate sharpness
            face_crop = img[y1:y2, x1:x2]
            sharpness = self.calculate_sharpness(face_crop)

            # Extract and normalize embedding
            raw_emb = face.embedding.astype(np.float32)
            norm = np.linalg.norm(raw_emb)
            normed_emb = (raw_emb / norm) if norm > 0 else raw_emb

            # Pose angles
            pitch, yaw, roll = [float(a) for a in face.pose]
            pose_label = self.classify_pose(pitch, yaw, roll)

            processed_faces.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_score": float(face.det_score),
                "landmarks": face.kps.tolist() if hasattr(face, "kps") else [],
                "pose": [pitch, yaw, roll],
                "pose_label": pose_label,
                "embedding": raw_emb,
                "normed_embedding": normed_emb,
                "sharpness": sharpness,
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            })

        return processed_faces

    @staticmethod
    def calculate_sharpness(image_crop: np.ndarray) -> float:
        """Calculates blurriness using variance of Laplacian. Higher is sharper."""
        if image_crop is None or image_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def classify_pose(pitch: float, yaw: float, roll: float) -> str:
        """
        Classifies head pose into standard enrollment angles:
        'front', 'left', 'right', 'up', 'down'.
        """
        if abs(yaw) <= 15.0 and abs(pitch) <= 15.0:
            return "front"
        elif yaw <= -18.0:
            return "left"
        elif yaw >= 18.0:
            return "right"
        elif pitch <= -15.0:
            return "up"
        elif pitch >= 15.0:
            return "down"
        return "angled"

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates cosine similarity between two 512-D feature vectors."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def match_face(
        self,
        query_embedding: np.ndarray,
        enrolled_db: Dict[Any, List[np.ndarray]],
        threshold: Optional[float] = None
    ) -> Tuple[Optional[Any], float, bool]:
        """
        Matches a query face embedding against a database of enrolled student embeddings.
        enrolled_db: {student_id: [emb1, emb2, ...]}
        Returns:
            (matched_student_id, highest_similarity_score, is_match)
            If no student exceeds the threshold, returns (None, highest_similarity_score, False).
        """
        if threshold is None:
            threshold = self.recognition_threshold

        best_student_id = None
        best_score = -1.0

        for student_id, embeddings in enrolled_db.items():
            for emb in embeddings:
                score = self.cosine_similarity(query_embedding, emb)
                if score > best_score:
                    best_score = score
                    best_student_id = student_id

        if best_score >= threshold:
            return best_student_id, best_score, True
        return None, max(best_score, 0.0), False

    def extract_from_video(
        self,
        video_path: str,
        max_samples: int = 8,
        min_sharpness: float = 5.0,
        frame_stride: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Samples frames from a student's enrollment video, filters for high quality
        and diverse pose angles, and extracts 512-D embeddings.
        """
        if not os.path.isfile(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        all_detected = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                faces = self.detect_faces(frame)
                # Keep only detections with a single prominent face
                if len(faces) == 1:
                    face = faces[0]
                    if face["det_score"] >= self.det_thresh:
                        all_detected.append(face)

            frame_idx += 1

        cap.release()

        if not all_detected:
            return []

        # Filter by sharpness if candidates qualify, otherwise use all detected faces
        sharp_candidates = [f for f in all_detected if f["sharpness"] >= min_sharpness]
        candidates = sharp_candidates if sharp_candidates else all_detected

        # Sort candidates to prioritize diverse poses and highest sharpness
        selected = []
        covered_poses = set()

        for cand in sorted(candidates, key=lambda f: f["sharpness"], reverse=True):
            pose = cand["pose_label"]
            if pose not in covered_poses or len(selected) < max_samples:
                selected.append(cand)
                covered_poses.add(pose)
            if len(selected) >= max_samples:
                break

        return selected

    @staticmethod
    def encrypt_embedding(embedding: np.ndarray, fernet_key: str | bytes) -> bytes:
        """
        Encrypts a 512-D float32 numpy vector into an opaque Fernet binary payload.
        No plain-text facial vectors or raw photos are retained.
        """
        if isinstance(fernet_key, str):
            fernet_key = fernet_key.encode("utf-8")
        cipher = Fernet(fernet_key)
        raw_bytes = embedding.astype(np.float32).tobytes()
        return cipher.encrypt(raw_bytes)

    @staticmethod
    def decrypt_embedding(encrypted_bytes: bytes, fernet_key: str | bytes) -> np.ndarray:
        """
        Decrypts an opaque Fernet binary payload back into a 512-D float32 numpy vector.
        """
        if isinstance(fernet_key, str):
            fernet_key = fernet_key.encode("utf-8")
        cipher = Fernet(fernet_key)
        raw_bytes = cipher.decrypt(encrypted_bytes)
        return np.frombuffer(raw_bytes, dtype=np.float32)
