import cv2
import mediapipe as mp


class PoseDetector:
    """
    Klasa odpowiedzialna za detekcję pozy przy użyciu MediaPipe.

    Args:
            mode (bool): Specifies the static image mode.
                - False (default): Treats the input as a video stream. The model detects
                  the most prominent person in the first few frames, and then simply
                  tracks the landmarks, which is much faster and provides smoother results.
                - True: Treats every frame as an unrelated, individual image. Forces full
                  detection on every frame (slower, but necessary for static photos).

            complexity (int): Model complexity. Accepts values 0, 1, or 2.
                - 0: Fastest, but lowest accuracy.
                - 1: Balanced.
                - 2: Most accurate, but most CPU-intensive.
                In the context of the "Cyber Coach" project, where precision is key for
                evaluating technique (e.g., squat depth), value 2 is recommended if hardware allows.

            smooth_landmarks (bool): Enables temporal filtering to reduce jitter.
                If True, landmark movement will be smoother, which is crucial for
                visualizing exercises and calculating stable joint angles.
                Only works when mode=False.

            enable_segmentation (bool): Whether to generate a segmentation mask (background removal).
                Useful if you plan to replace the background behind the user, but it increases latency.

            smooth_segmentation (bool): Whether to filter the segmentation mask for smoother edges.
                Only effective if enable_segmentation=True.

            detection_con (float): Minimum detection confidence (0.0 - 1.0).
                The threshold for the model to consider the initial person detection successful.
                Higher values reduce false positives (e.g., detecting a coat rack as a person).

            track_con (float): Minimum tracking confidence (0.0 - 1.0).
                The threshold for the model to consider the tracked landmarks valid.
                If confidence drops below this, the model invokes full detection again.
                High values increase robustness against losing the pose during fast movements.
        """

    def __init__(self,
                 mode=False,
                 complexity=2,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 detection_con=0.5,
                 track_con=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.connection_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

        self.results = None

    def find_pose(self, img, draw=True):
        """
        Processes the image to find and optionally draw pose landmarks.
        :param img: single frame (BGR)
        :param draw: should we draw the landmarks on the image
        :return: processed frame
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=self.connection_style
            )

        return img

    def get_landmarks(self):
        """
        Returns raw landmark data (keypoints) detected by MediaPipe.

        Returns:
            NormalizedLandmarkList or None: An object containing a list of 33 keypoints,
            or None if no person is detected.

            Access specific points via index (e.g., landmarks[11] is the left shoulder).
            Each individual landmark in this list possesses 4 attributes:

            - x (float): Horizontal coordinate normalized to [0.0, 1.0].
              Multiply this by the image width to get the pixel position.

            - y (float): Vertical coordinate normalized to [0.0, 1.0].
              Multiply this by the image height to get the pixel position.

            - z (float): Depth relative to the midpoint of the hips.
              Values are on a scale similar to x and y.
              * Negative value: The point is closer to the camera than the hips.
              * Positive value: The point is further away from the camera.

            - visibility (float): Probability (0.0 - 1.0) that the landmark is visible
              (not occluded by another body part) and present in the frame.
              It is recommended to filter out points with low visibility to prevent errors.
        """

        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks
        return None


