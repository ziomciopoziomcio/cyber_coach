import math
from typing import Optional, Tuple, Dict, Sequence, Any
import numpy as np


_MP_IDX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class JointAngleCalculator:
    """
    Calculates joint angles based on MediaPipe landmarks.

    Parameters
    ----------
    visibility_threshold : float
        Minimum value of the `visibility` field for a landmark to be used.

    Usage
    -----
    calc = JointAngleCalculator(visibility_threshold=0.5)
    angle = calc.get_joint_angle(landmarks, "left_knee", image_shape)
    """

    def __init__(self, visibility_threshold: float = 0.5):
        self.visibility_threshold = visibility_threshold

    @staticmethod
    def _image_hw(image_shape: Tuple[int, ...]) -> Tuple[int, int]:
        """
        Accepts (h, w) or (h, w, c) and returns (h, w).
        """
        if len(image_shape) >= 2:
            return int(image_shape[0]), int(image_shape[1])
        raise ValueError("image_shape musi mieć co najmniej 2 elementy: (h, w [, c])")

    def _landmark_to_point(self, lm: Any, image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Converts a single MediaPipe landmark (has x, y and optional visibility)
        to pixel coordinates [x, y]. Returns None when the point is missing
        or visibility is below the threshold.
        """
        if lm is None:
            return None
        h, w = self._image_hw(image_shape)
        vis = getattr(lm, "visibility", None)
        if vis is not None and vis < self.visibility_threshold:
            return None
        # Zakładamy, że lm.x i lm.y są znormalizowane [0,1]
        return np.array([float(lm.x) * w, float(lm.y) * h], dtype=float)

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        """
        Angle (in degrees) at vertex b formed by points a-b-c.
        Returns None when any vector has zero length.
        """
        ba = a - b
        bc = c - b
        na = np.linalg.norm(ba)
        nb = np.linalg.norm(bc)
        if na == 0 or nb == 0:
            return None
        cos_angle = float(np.clip(np.dot(ba, bc) / (na * nb), -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))

    def _landmarks_list(self, landmarks: Any) -> Optional[Sequence]:
        """
        Returns a list-like object of landmarks (MediaPipe NormalizedLandmarkList or a list)
        or None.
        """
        if landmarks is None:
            return None
        try:
            return landmarks.landmark  # MediaPipe NormalizedLandmarkList
        except Exception:
            return landmarks  # lista/iterowalny

    def get_joint_angle(self, landmarks: Any, joint: str, image_shape: Tuple[int, ...]) -> Optional[float]:
        """
        Returns the angle (in degrees) for the specified joint, e.g. 'left_elbow' or 'right_knee'.
        Returns None when points are missing or their visibility is too low.
        """
        joint = joint.lower()
        chains = {
            "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
            "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
            "left_knee": ("left_hip", "left_knee", "left_ankle"),
            "right_knee": ("right_hip", "right_knee", "right_ankle"),
            "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
            "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
            "left_hip": ("left_shoulder", "left_hip", "left_knee"),
            "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        }

        if joint not in chains:
            return None

        lm_list = self._landmarks_list(landmarks)
        if lm_list is None:
            return None

        a_name, b_name, c_name = chains[joint]
        try:
            a_idx = _MP_IDX[a_name]
            b_idx = _MP_IDX[b_name]
            c_idx = _MP_IDX[c_name]
        except KeyError:
            return None

        a_lm = lm_list[a_idx] if len(lm_list) > a_idx else None
        b_lm = lm_list[b_idx] if len(lm_list) > b_idx else None
        c_lm = lm_list[c_idx] if len(lm_list) > c_idx else None

        a_pt = self._landmark_to_point(a_lm, image_shape)
        b_pt = self._landmark_to_point(b_lm, image_shape)
        c_pt = self._landmark_to_point(c_lm, image_shape)

        if a_pt is None or b_pt is None or c_pt is None:
            return None

        return self._angle_between(a_pt, b_pt, c_pt)

    def get_all_angles(self, landmarks: Any, image_shape: Tuple[int, ...]) -> Dict[str, Optional[float]]:
        """
        Returns a dictionary of angles for commonly used joints.
        """
        keys = [
            "left_elbow", "right_elbow",
            "left_knee", "right_knee",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
        ]
        return {k: self.get_joint_angle(landmarks, k, image_shape) for k in keys}