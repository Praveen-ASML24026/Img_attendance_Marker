import os
import cv2
import base64
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from app.face_engine.engine import FaceEngine


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculates Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0, x2 - x1)
    intersection_h = max(0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0

    return float(intersection_area / union_area)


def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.40) -> List[Dict[str, Any]]:
    """
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate detections
    resulting from overlapping quadrant tiles.
    """
    if not detections:
        return []

    # Sort detections by detection confidence score descending
    sorted_dets = sorted(detections, key=lambda d: d.get("det_score", 0.0), reverse=True)
    kept = []

    for candidate in sorted_dets:
        box_candidate = candidate["bbox"]
        should_keep = True

        for accepted in kept:
            box_accepted = accepted["bbox"]
            iou = calculate_iou(box_candidate, box_accepted)
            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept.append(candidate)

    return kept


class ClassroomProcessor:
    """
    High-density classroom multi-face detection, tiling, and recognition processor.
    Designed for 60+ student classroom environments where back-row faces may be small.
    """

    def __init__(self, engine: Optional[FaceEngine] = None, recognition_threshold: float = 0.50):
        self.engine = engine if engine is not None else FaceEngine(recognition_threshold=recognition_threshold)
        self.recognition_threshold = recognition_threshold

    def detect_classroom_faces(
        self,
        image_input: Any,
        use_tiling: bool = True,
        grid_rows: int = 2,
        grid_cols: int = 2,
        overlap: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Detects all faces in a classroom photograph using multi-scale inference.
        Combines a full-image scan with an overlapping quadrant tiling pass
        to resolve small faces in middle and back rows.
        """
        image = self.engine.load_image(image_input)
        if image is None:
            return []

        h, w = image.shape[:2]
        all_detections = []

        # Pass 1: Global Full-Frame Detection
        global_faces = self.engine.detect_faces(image)
        for face in global_faces:
            face["source"] = "global"
            all_detections.append(face)

        # Pass 2: Overlapping Quadrant Tiling (for dense/small back-row faces)
        if use_tiling and w >= 640 and h >= 480:
            tile_w = int(w / grid_cols * (1.0 + overlap))
            tile_h = int(h / grid_rows * (1.0 + overlap))

            step_x = int((w - tile_w) / max(grid_cols - 1, 1)) if grid_cols > 1 else 0
            step_y = int((h - tile_h) / max(grid_rows - 1, 1)) if grid_rows > 1 else 0

            for r in range(grid_rows):
                for c in range(grid_cols):
                    x1 = min(c * step_x, w - tile_w) if grid_cols > 1 else 0
                    y1 = min(r * step_y, h - tile_h) if grid_rows > 1 else 0
                    x2 = min(x1 + tile_w, w)
                    y2 = min(y1 + tile_h, h)

                    tile = image[y1:y2, x1:x2]
                    tile_faces = self.engine.detect_faces(tile)

                    # Re-project tile-local coordinates to global image space
                    for tf in tile_faces:
                        tx1, ty1, tx2, ty2 = tf["bbox"]
                        global_bbox = [
                            int(tx1 + x1),
                            int(ty1 + y1),
                            int(tx2 + x1),
                            int(ty2 + y1)
                        ]
                        tf["bbox"] = global_bbox
                        tf["source"] = f"tile_{r}_{c}"

                        # Also re-project landmarks if present
                        if "landmarks" in tf and tf["landmarks"]:
                            tf["landmarks"] = [
                                [lmk[0] + x1, lmk[1] + y1] for lmk in tf["landmarks"]
                            ]

                        all_detections.append(tf)

        # Pass 3: Non-Maximum Suppression to deduplicate overlapping boxes
        unique_faces = apply_nms(all_detections, iou_threshold=0.40)
        return unique_faces

    @staticmethod
    def crop_face_base64(image: np.ndarray, bbox: List[int], padding_percent: float = 0.20) -> str:
        """
        Crops face region with contextual padding and encodes to base64 JPEG
        for frontend review cards.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = int(box_w * padding_percent)
        pad_y = int(box_h * padding_percent)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return ""

        # Resize thumbnail to standard size for fast browser rendering
        thumbnail = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", thumbnail, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer).decode("utf-8")

    def process_session_photos(
        self,
        photo_inputs: List[Any],
        enrolled_db: Dict[str, List[np.ndarray]],
        student_metadata: Optional[Dict[str, str]] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes 2–3 classroom photographs for an attendance session:
        1. Runs multi-scale tiled detection.
        2. Matches faces against enrolled student embeddings via cosine similarity.
        3. Cross-photo deduplication: aggregates recognitions across all photos,
           retaining highest similarity score.
        4. Identifies unknown faces with thumbnail crops.
        5. Identifies absent/missing enrolled students.
        """
        if threshold is None:
            threshold = self.recognition_threshold

        if student_metadata is None:
            student_metadata = {}

        recognized_students: Dict[str, Dict[str, Any]] = {}
        unknown_faces: List[Dict[str, Any]] = []
        total_faces_count = 0

        for photo_idx, photo_input in enumerate(photo_inputs):
            image = self.engine.load_image(photo_input)
            if image is None:
                continue

            detected_faces = self.detect_classroom_faces(image, use_tiling=True)
            total_faces_count += len(detected_faces)

            for face_idx, face in enumerate(detected_faces):
                query_emb = face["normed_embedding"]

                matched_id, score, is_match = self.engine.match_face(
                    query_emb,
                    enrolled_db,
                    threshold=threshold
                )

                if is_match and matched_id is not None:
                    # Cross-photo aggregation: keep the highest confidence match
                    if matched_id not in recognized_students or score > recognized_students[matched_id]["confidence"]:
                        recognized_students[matched_id] = {
                            "student_id": matched_id,
                            "name": student_metadata.get(matched_id, f"Student {matched_id}"),
                            "confidence": round(float(score), 4),
                            "photo_index": photo_idx + 1,
                            "bbox": face["bbox"],
                            "pose": face["pose_label"],
                            "status": "Present"
                        }
                else:
                    # Unknown face detection
                    crop_b64 = self.crop_face_base64(image, face["bbox"])
                    unknown_faces.append({
                        "unknown_id": f"UNK_{photo_idx + 1}_{face_idx + 1}",
                        "photo_index": photo_idx + 1,
                        "bbox": face["bbox"],
                        "confidence": round(float(face["det_score"]), 4),
                        "best_similarity": round(float(score), 4),
                        "best_candidate_id": matched_id,
                        "thumbnail_base64": crop_b64
                    })

        # Identify missing (unrecognized) enrolled students
        missing_students = []
        for stu_id, name in student_metadata.items():
            if stu_id not in recognized_students:
                missing_students.append({
                    "student_id": stu_id,
                    "name": name,
                    "status": "Absent (Unconfirmed)"
                })

        return {
            "photos_count": len(photo_inputs),
            "total_detections": total_faces_count,
            "recognized_count": len(recognized_students),
            "unknown_count": len(unknown_faces),
            "missing_count": len(missing_students),
            "recognized_students": recognized_students,
            "unknown_faces": unknown_faces,
            "missing_students": missing_students
        }

