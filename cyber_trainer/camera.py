from posedetector import PoseDetector
from preprocessing import JointAngleCalculator
from pathlib import Path
import sys
import cv2
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))
from analysis.exercise_rules import ShoulderPressRules, JointStatus


def main():
    USE_CAMERA = False
    ENABLE_DUAL_VIEW = True
    VIEW_TYPE = 'side'
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

    last_rep_messages = [None] * len(caps)
    last_rep_times = [0] * len(caps)
    MESSAGE_DURATION = 3.0

    confirmed_reps = 0

    print(f"Tryb: {'Oba widoki (synchronizacja)' if ENABLE_DUAL_VIEW else VIEW_TYPE}")
    print(f"Źródło: {'Kamery na żywo' if USE_CAMERA else 'Pliki wideo'}")
    print("Naciśnij 'q' aby zakończyć\n")

    while True:
        frames = []
        all_ended = True

        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                all_ended = False
            else:
                frames.append(None)

        if all_ended:
            print("Koniec wideo.")
            break

        angles_front = None
        angles_side = None

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

                if view_name == 'front':
                    angles_front = angles
                elif view_name == 'side':
                    angles_side = angles

                has_errors = rule_set.has_angle_errors(angles) if ENABLE_FEEDBACK else False

                if ENABLE_DUAL_VIEW and view_name == 'front':
                    completed_rep = rule_set.update_repetition_tracking(angles, frame_idx)

                    if completed_rep:
                        side_has_errors = False
                        if angles_side is not None:
                            side_has_errors = rules_side.has_angle_errors(angles_side)

                        if completed_rep.is_complete and not side_has_errors:
                            confirmed_reps += 1
                            status = "ZALICZONE"
                            color = COLOR_OK
                            print(f"  Powtórzenie #{confirmed_reps} ZALICZONE")
                            print(f"  ROM: {completed_rep.rom:.1f}°")
                        else:
                            status = "ODRZUCONE"
                            color = COLOR_ERROR
                            if not completed_rep.is_complete:
                                print(f"Front: {' | '.join(completed_rep.errors)}")
                            if side_has_errors:
                                print(f"Side: Biodro poza zakresem")

                        for j in range(len(caps)):
                            last_rep_messages[j] = (status, color, completed_rep.rom)
                            last_rep_times[j] = time.time()

                elif not ENABLE_DUAL_VIEW:
                    completed_rep = rule_set.update_repetition_tracking(angles, frame_idx)

                    if completed_rep:
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

                for joint_name, angle in angles.items():
                    if angle is None:
                        continue
                    if joint_name not in ANGLE_TO_IDX:
                        continue

                    idx = ANGLE_TO_IDX[joint_name]
                    try:
                        lm = landmarks.landmark[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        if ENABLE_FEEDBACK:
                            status = rule_set.check_angles(angles).get(joint_name)
                            if status == JointStatus.ERROR:
                                color = COLOR_ERROR
                            elif status == JointStatus.OK:
                                color = COLOR_OK
                            else:
                                color = COLOR_NEUTRAL
                        else:
                            color = COLOR_NEUTRAL

                        cv2.putText(frame, f"{int(angle)}", (x + 15, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(frame, (x, y), 8, color, -1)
                    except (IndexError, AttributeError):
                        continue

            # HUD
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time

            # Czarne tło pod górnym HUD
            cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Powtorzenia: {confirmed_reps}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OK, 2)

            # Komunikat o ostatnim powtórzeniu - teraz u góry
            if last_rep_messages[i] is not None:
                if (time.time() - last_rep_times[i]) < MESSAGE_DURATION:
                    status_msg, msg_color, rom = last_rep_messages[i]
                    cv2.putText(frame, f"{status_msg} | ROM: {rom:.1f} deg", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

            cv2.imshow(window_name, frame)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nZakończono przez użytkownika")
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    print("\n=== PODSUMOWANIE ===")
    for i, (rule_set, view_name) in enumerate(zip(rules_list, view_names)):
        summary = rule_set.get_repetition_summary()
        print(f"\n{view_name.upper()} VIEW:")
        print(f"  Wszystkie wykryte: {summary['total_reps']}")
        print(f"  Prawidłowe: {summary['complete_reps']}")
        print(f"  Nieprawidłowe: {summary['incomplete_reps']}")
        if summary['complete_reps'] > 0:
            print(f"  Średni ROM: {summary['avg_rom']:.1f}°")

    if ENABLE_DUAL_VIEW:
        print(f"\nZATWIERDZONE (oba widoki OK): {confirmed_reps}")


if __name__ == '__main__':
    main()