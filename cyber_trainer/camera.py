from posedetector import PoseDetector
from preprocessing import JointAngleCalculator
from pathlib import Path
import sys
import cv2
import time

# Dodaj ścieżkę do modułu analysis
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
from analysis.exercise_rules import ShoulderPressRules, JointStatus


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

    window_name = 'Cyber Coach - Live Training'

    cap = cv2.VideoCapture(source)
    detector = PoseDetector(complexity=2)
    calc = JointAngleCalculator(visibility_threshold=0.5)
    rules = ShoulderPressRules(view_type=VIEW_TYPE)

    p_time = 0
    frame_idx = 0

    ANGLE_TO_IDX = {
        "left_elbow": 13, "right_elbow": 14,
        "left_knee": 25, "right_knee": 26,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24
    }

    # Kolory
    COLOR_OK = (0, 255, 0)  # zielony
    COLOR_ERROR = (0, 0, 255)  # czerwony
    COLOR_NEUTRAL = (200, 200, 200)  # szary
    COLOR_WARNING = (0, 165, 255)  # pomarańczowy

    # Stan dla wyświetlania ostatniego powtórzenia
    last_rep_message = None
    last_rep_time = 0
    MESSAGE_DURATION = 3.0  # sekundy

    print(f"🎯 Tryb: {VIEW_TYPE}")
    if ENABLE_FEEDBACK:
        print("🟢 = kąt OK | 🔴 = kąt niepoprawny | ⚪ = punkt niewidoczny")
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

            # Sprawdź błędy (TYLKO widoczne kąty poza zakresem)
            has_errors = rules.has_angle_errors(angles) if ENABLE_FEEDBACK else False

            # Update repetition tracking
            completed_rep = rules.update_repetition_tracking(angles, frame_idx)

            # Jeśli zakończono powtórzenie, zapisz komunikat
            if completed_rep:
                if completed_rep.is_complete:
                    status = "DOBRE POWTORZENIE"
                    color = COLOR_OK
                    print(f"✓ Rep {len(rules.repetitions)}: ROM={completed_rep.rom:.1f}°")
                else:
                    status = "ODRZUCONE"  # ← zmieniono z "NIEPEŁNE"
                    color = COLOR_ERROR
                    # Wyświetl powody odrzucenia
                    error_msg = " | ".join(completed_rep.errors)
                    print(f"✗ Rep {len(rules.repetitions)}: {error_msg}")

                last_rep_message = (status, color, completed_rep.rom)
                last_rep_time = time.time()

            # Pobierz status każdego stawu
            angle_status = rules.check_angles(angles) if ENABLE_FEEDBACK else {}

            # Rysuj kąty
            for name, angle in angles.items():
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

                # Wybierz kolor na podstawie statusu
                status = angle_status.get(name, JointStatus.MISSING)

                if status == JointStatus.OK:
                    color = COLOR_OK
                    angle_text = f'{int(angle)} deg' if angle else '?'
                elif status == JointStatus.ERROR:
                    color = COLOR_ERROR
                    angle_text = f'{int(angle)} deg!' if angle else '?'
                else:  # MISSING
                    color = COLOR_NEUTRAL
                    angle_text = '-'

                # Rysuj punkt i kąt
                cv2.circle(frame, (x, y), 6, color, -1)
                label = f'{name.split("_")[0].capitalize()} {angle_text}'
                cv2.putText(frame, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Górny status - błędy techniki
            if ENABLE_FEEDBACK:
                if has_errors:
                    status_text = "NIEPOPRAWNE CWICZENIE"
                    status_color = COLOR_ERROR
                else:
                    status_text = "OK"
                    status_color = COLOR_OK

                cv2.rectangle(frame, (10, 10), (w - 10, 60), (0, 0, 0), -1)
                cv2.putText(frame, status_text, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

            # Licznik powtórzeń (prawy górny róg)
            summary = rules.get_repetition_summary()
            rep_text = f"Powtorzenia: {summary['complete_reps']}"
            cv2.putText(frame, rep_text, (w - 300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Wyświetl komunikat o ostatnim powtórzeniu (przez 3 sekundy)
            if last_rep_message and (time.time() - last_rep_time) < MESSAGE_DURATION:
                msg_text, msg_color, rom = last_rep_message

                # Tło dla komunikatu
                msg_y = 100
                cv2.rectangle(frame, (10, msg_y), (w - 10, msg_y + 80), (0, 0, 0), -1)

                # Główny komunikat
                cv2.putText(frame, msg_text, (20, msg_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, msg_color, 3, cv2.LINE_AA)

                # ROM info
                rom_text = f"ROM: {rom:.1f} deg (min: {rules.MIN_ROM} deg)"
                cv2.putText(frame, rom_text, (20, msg_y + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, h - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_idx += 1

    # Podsumowanie po zakończeniu
    print("\n" + "=" * 50)
    print("PODSUMOWANIE TRENINGU:")
    summary = rules.get_repetition_summary()
    print(f"Łącznie powtórzeń: {summary['total_reps']}")
    print(f"Pełnych: {summary['complete_reps']}")
    print(f"Niepełnych: {summary['incomplete_reps']}")
    if summary['total_reps'] > 0:
        print(f"Średni ROM: {summary['avg_rom']:.1f}°")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
