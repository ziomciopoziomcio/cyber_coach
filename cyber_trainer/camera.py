# cyber_trainer/camera.py
from posedetector import PoseDetector
from preprocessing import JointAngleCalculator
from pathlib import Path
import sys
import cv2
import time

# Dodaj ścieżkę do modułu analysis
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
from analysis.exercise_rules import ShoulderPressRules


def main():
    # Konfiguracja
    USE_CAMERA = False  # True = kamera, False = plik wideo
    VIEW_TYPE = 'side'  # 'front' lub 'side'
    ENABLE_FEEDBACK = True  # włącz/wyłącz kolorowy feedback

    if USE_CAMERA:
        source = 0
    else:
        project_root = Path(__file__).parent.parent
        source = str(project_root / 'data' / 'videos' / 'fail' / 'jurek_4_bok.mp4')

    window_name = 'Cyber Coach - Camera Test Preview'

    cap = cv2.VideoCapture(source)
    detector = PoseDetector(complexity=2)
    calc = JointAngleCalculator(visibility_threshold=0.5)
    rules = ShoulderPressRules(view_type=VIEW_TYPE)

    p_time = 0

    ANGLE_TO_IDX = {
        "left_elbow": 13, "right_elbow": 14,
        "left_knee": 25, "right_knee": 26,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24
    }

    print(f"🎯 Tryb: {VIEW_TYPE}")
    if ENABLE_FEEDBACK:
        print("🟢 = kąt OK | 🔴 = kąt źle")
    print("Naciśnij 'q' aby zakończyć")

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

            # Sprawdź poprawność kątów (jeśli feedback włączony)
            if ENABLE_FEEDBACK:
                angle_status = rules.check_angles(angles)
                pose_correct = rules.is_pose_correct(angles)
            else:
                angle_status = {name: True for name in angles.keys()}
                pose_correct = True

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

                # Kolor na podstawie poprawności (zielony=OK, czerwony=błąd)
                is_correct = angle_status.get(name, True)
                color = (0, 255, 0) if is_correct else (0, 0, 255)

                cv2.circle(frame, (x, y), 6, color, -1)
                label = f'{name.split("_")[0].capitalize()} {int(angle)} deg'
                cv2.putText(frame, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # Status całej pozy (góra ekranu)
            if ENABLE_FEEDBACK:
                status_text = "POPRAWNA POZA" if pose_correct else "NIEPOPRAWNA POZA"
                status_color = (0, 255, 0) if pose_correct else (0, 0, 255)

                cv2.rectangle(frame, (10, 10), (w - 10, 60), (0, 0, 0), -1)
                cv2.putText(frame, status_text, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, h - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
