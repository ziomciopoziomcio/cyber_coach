from posedetector import PoseDetector
from preprocessing import JointAngleCalculator
from pathlib import Path
import sys
import cv2
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
from analysis.exercise_rules import ShoulderPressRules, JointStatus


def main():
    USE_CAMERA = False  # True = kamery na żywo, False = pliki wideo
    ENABLE_DUAL_VIEW = True  # True = przód + bok jednocześnie, False = jeden widok
    VIEW_TYPE = 'side'  # tylko jeśli ENABLE_DUAL_VIEW = False
    ENABLE_FEEDBACK = True

    project_root = Path(__file__).parent.parent

    if ENABLE_DUAL_VIEW:
        if USE_CAMERA:
            source_front = 0
            source_side = 1
        else:
            source_front = str(project_root / 'data' / 'videos' / 'try2' / 'nina_1_przod.mp4')
            source_side = str(project_root / 'data' / 'videos' / 'try2' / 'nina_1_bok.mp4')

        cap_front = cv2.VideoCapture(source_front)
        cap_side = cv2.VideoCapture(source_side)

        rules_front = ShoulderPressRules(view_type='front')
        rules_side = ShoulderPressRules(view_type='side')

        caps = [cap_front, cap_side]
        rules_list = [rules_front, rules_side]
        window_names = ['Front View', 'Side View']
        view_names = ['front', 'side']
    else:
        source = 0 if USE_CAMERA else str(project_root / 'data' / 'videos' / 'try1' / 'jurek_1_bok.mp4')
        cap = cv2.VideoCapture(source)
        rules_single = ShoulderPressRules(view_type=VIEW_TYPE)

        caps = [cap]
        rules_list = [rules_single]
        window_names = ['Cyber Coach - Live Training']
        view_names = [VIEW_TYPE]

    detector = PoseDetector(complexity=2)
    calc = JointAngleCalculator(visibility_threshold=0.5)

    p_time = 0
    frame_idx = 0

    ANGLE_TO_IDX = {
        "left_elbow": 13, "right_elbow": 14,
        "left_knee": 25, "right_knee": 26,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24
    }

    COLOR_OK = (0, 255, 0)
    COLOR_ERROR = (0, 0, 255)
    COLOR_NEUTRAL = (200, 200, 200)

    # Przechowujemy informacje o ostatnich powtórzeniach z każdego widoku
    last_completed_reps = [None] * len(caps)  # ostatnie zakończone rep z każdego widoku
    last_rep_messages = [None] * len(caps)  # komunikaty dla każdego okna
    last_rep_times = [0] * len(caps)
    MESSAGE_DURATION = 3.0

    # Globalny licznik ZATWIERDZONYCH powtórzeń (oba widoki OK)
    confirmed_reps = 0

    print(f"Tryb: {'Oba widoki (synchronizacja)' if ENABLE_DUAL_VIEW else VIEW_TYPE}")
    print(f"Źródło: {'Kamery na żywo' if USE_CAMERA else 'Pliki wideo'}")
    print("Naciśnij 'q' aby zakończyć\n")

    while True:
        frames = []
        all_ended = True

        # Odczyt klatek ze wszystkich źródeł
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                all_ended = False
                frames.append(frame)
            else:
                frames.append(None)

        if all_ended:
            print("Koniec wideo.")
            break

        # Przetwarzaj każdy widok
        for i, (frame, rule_set, window_name, view_name) in enumerate(
                zip(frames, rules_list, window_names, view_names)):
            if frame is None:
                continue

            frame = detector.find_pose(frame, draw=True)
            landmarks = detector.get_landmarks()
            h, w = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1

            if landmarks:
                angles = calc.get_all_angles(landmarks, (h, w, channels))
                has_errors = rule_set.has_angle_errors(angles) if ENABLE_FEEDBACK else False

                # Detekcja powtórzenia w tym widoku
                completed_rep = rule_set.update_repetition_tracking(angles, frame_idx)

                if completed_rep:
                    last_completed_reps[i] = completed_rep

                    if ENABLE_DUAL_VIEW:
                        # Sprawdź czy oba widoki mają zakończone powtórzenie
                        front_rep = last_completed_reps[0]
                        side_rep = last_completed_reps[1]

                        # Jeśli oba widoki zakończyły powtórzenie (zbliżone w czasie)
                        if front_rep and side_rep:
                            # Sprawdź czy były w podobnym czasie (tolerance 30 klatek)
                            frame_diff = abs(front_rep.end_frame - side_rep.end_frame)

                            if frame_diff < 30:
                                # OBA WIDOKI ZAKOŃCZONE - teraz sprawdź czy oba są OK
                                both_valid = front_rep.is_complete and side_rep.is_complete

                                if both_valid:
                                    confirmed_reps += 1
                                    status = "ZALICZONE"
                                    color = COLOR_OK
                                    print(f"  Powtórzenie #{confirmed_reps} ZALICZONE")
                                    print(f"  Front ROM: {front_rep.rom:.1f}°")
                                    print(f"  Side ROM: {side_rep.rom:.1f}°")
                                else:
                                    status = "✗ ODRZUCONE"
                                    color = COLOR_ERROR
                                    print(f"✗ Powtórzenie ODRZUCONE:")
                                    if not front_rep.is_complete:
                                        print(f"  Front: {' | '.join(front_rep.errors)}")
                                    if not side_rep.is_complete:
                                        print(f"  Side: {' | '.join(side_rep.errors)}")

                                # Zapisz komunikat dla OBU okien
                                for j in range(len(caps)):
                                    rep = last_completed_reps[j]
                                    last_rep_messages[j] = (status, color, rep.rom if rep else 0)
                                    last_rep_times[j] = time.time()

                                # Wyczyść bufory
                                last_completed_reps[0] = None
                                last_completed_reps[1] = None
                    else:
                        # Tryb pojedynczego widoku (stara logika)
                        if completed_rep.is_complete:
                            confirmed_reps += 1
                            status = "DOBRE POWTORZENIE"
                            color = COLOR_OK
                            print(f"✓ Rep {confirmed_reps}: ROM={completed_rep.rom:.1f}°")
                        else:
                            status = "ODRZUCONE"
                            color = COLOR_ERROR
                            error_msg = " | ".join(completed_rep.errors)
                            print(f"Rep: {error_msg}")

                        last_rep_messages[i] = (status, color, completed_rep.rom)
                        last_rep_times[i] = time.time()

                angle_status = rule_set.check_angles(angles) if ENABLE_FEEDBACK else {}

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

                    status = angle_status.get(name, JointStatus.MISSING)

                    if status == JointStatus.OK:
                        color = COLOR_OK
                        angle_text = f'{int(angle)} deg' if angle else '?'
                    elif status == JointStatus.ERROR:
                        color = COLOR_ERROR
                        angle_text = f'{int(angle)} deg!' if angle else '?'
                    else:
                        color = COLOR_NEUTRAL
                        angle_text = '-'

                    cv2.circle(frame, (x, y), 6, color, -1)
                    label = f'{name.split("_")[0].capitalize()} {angle_text}'
                    cv2.putText(frame, label, (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                # Status techniczny
                if ENABLE_FEEDBACK:
                    status_text = "NIEPOPRAWNE CWICZENIE" if has_errors else "OK"
                    status_color = COLOR_ERROR if has_errors else COLOR_OK

                    cv2.rectangle(frame, (10, 10), (w - 10, 60), (0, 0, 0), -1)
                    cv2.putText(frame, status_text, (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

                # Licznik - ZSYNCHRONIZOWANY dla dual view
                rep_count = confirmed_reps if ENABLE_DUAL_VIEW else rule_set.get_repetition_summary()['complete_reps']
                rep_text = f"Powtorzenia: {rep_count}"
                cv2.putText(frame, rep_text, (w - 300, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # Komunikat o ostatnim powtórzeniu
                if last_rep_messages[i] and (time.time() - last_rep_times[i]) < MESSAGE_DURATION:
                    msg_text, msg_color, rom = last_rep_messages[i]
                    msg_y = 100

                    cv2.rectangle(frame, (10, msg_y), (w - 10, msg_y + 80), (0, 0, 0), -1)
                    cv2.putText(frame, msg_text, (20, msg_y + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, msg_color, 3, cv2.LINE_AA)

                    rom_text = f"ROM: {rom:.1f} deg (min: {rule_set.MIN_ROM} deg)"
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

    print("\n" + "=" * 50)
    print("PODSUMOWANIE TRENINGU:")
    if ENABLE_DUAL_VIEW:
        print(f"\nZATWIERDZONYCH POWTÓRZEŃ: {confirmed_reps}")
        print("\nSzczegóły per widok:")
        for rule_set, view_name in zip(rules_list, view_names):
            summary = rule_set.get_repetition_summary()
            print(f"\n{view_name.upper()}:")
            print(f"  Wykrytych cykli: {summary['total_reps']}")
            print(f"  Poprawnych technicznie: {summary['complete_reps']}")
            if summary['total_reps'] > 0:
                print(f"  Średni ROM: {summary['avg_rom']:.1f}°")
    else:
        summary = rules_list[0].get_repetition_summary()
        print(f"Łącznie: {summary['total_reps']}")
        print(f"Pełnych: {summary['complete_reps']}")
        print(f"Niepełnych: {summary['incomplete_reps']}")
        if summary['total_reps'] > 0:
            print(f"Średni ROM: {summary['avg_rom']:.1f}°")

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()