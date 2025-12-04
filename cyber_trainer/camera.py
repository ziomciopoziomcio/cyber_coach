from posedetector import PoseDetector
from preprocessing import JointAngleCalculator
import cv2
import time

def main():

    source = 0 # source config
    window_name = 'Cyber Coach - Camera Test Preview'

    cap = cv2.VideoCapture(source)
    detector = PoseDetector(complexity=2)
    calc = JointAngleCalculator(visibility_threshold=0.5)
    p_time = 0

    ANGLE_TO_IDX = {
        "left_elbow": 13, "right_elbow": 14,
        "left_knee": 25, "right_knee": 26,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error or end of video.")
            break

        frame = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        h, w = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1

        if landmarks:
            angles = calc.get_all_angles(landmarks, (h, w, channels))

            for name, angle in angles.items():
                if angle is None:
                    continue
                idx = ANGLE_TO_IDX.get(name)
                if idx is None:
                    continue
                try:
                    lm = landmarks.landmark[idx] if hasattr(landmarks, "landmark") else landmarks[idx]
                    vis = getattr(lm, "visibility", None)
                    if vis is not None and vis < calc.visibility_threshold:
                        continue
                    x, y = int(lm.x * w), int(lm.y * h)
                except Exception:
                    continue

                cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)
                label = f'{name.split("_")[0].capitalize()} {int(angle)} deg'
                cv2.putText(frame, label, (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS Calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()