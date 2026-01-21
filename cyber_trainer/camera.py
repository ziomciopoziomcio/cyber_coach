from posedetector import PoseDetector
from cyber_trainer.preprocessing import JointAngleCalculator
from components.phone_camera import IPWebcamClient
from analysis.exercise_rules import ShoulderPressRules, JointStatus
from pathlib import Path
import sys
import cv2
import time
import threading
import re

import logging
logger = logging.getLogger(__name__)

# new imports for speech-to-text
from components.speech_to_text import start_listening, stop_listening

sys.path.insert(0, str(Path(__file__).parent.parent))

WAIT_FIRST_FRAME = 5.0
POLL_INTERVAL = 0.05
SYNC_FRAME_THRESHOLD = 300

def wait_for_first_frame(client, timeout=WAIT_FIRST_FRAME, poll=POLL_INTERVAL):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if client.get_current_frame() is not None:
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False

def main():
    use_camera = False
    use_phone_streams = False
    enable_dual_view = True
    view_type = 'front'
    enable_feedback = True

    project_root = Path(__file__).parent.parent

    phone_clients = []
    caps = []
    rules_list = []
    window_names = []
    view_names = []

    # Konfiguracja źródeł i reguł
    if enable_dual_view:
        # reguły i nazwy okien dla obu widoków (zawsze ustawiamy)
        rules_front = ShoulderPressRules(view_type='front')
        rules_side = ShoulderPressRules(view_type='side')
        rules_list = [rules_front, rules_side]
        window_names = ['Front View', 'Side View']
        view_names = ['front', 'side']

        if use_phone_streams:
            phone_front_url = "http://192.168.1.237:8081"
            phone_side_url = "http://192.168.1.101:8080" # do uzupełnienia

            client_front = IPWebcamClient(phone_front_url)
            client_side = IPWebcamClient(phone_side_url)
            client_front.start_stream()
            client_side.start_stream()
            if not wait_for_first_frame(client_front):
                logger.warning("Nie otrzymano pierwszej klatki z front telefonu w ciągu kilku sekund.")
            if not wait_for_first_frame(client_side):
                logger.warning("Nie otrzymano pierwszej klatki z side telefonu w ciągu kilku sekund.")
            phone_clients = [client_front, client_side]
            caps = [None, None]
        else:
            source_front = 0 if use_camera else str(project_root / 'data' / 'videos' / 'try2' / 'nina_1_przod.mp4')
            source_side = 1 if use_camera else str(project_root / 'data' / 'videos' / 'try2' / 'nina_1_bok.mp4')
            cap_front = cv2.VideoCapture(source_front)
            cap_side = cv2.VideoCapture(source_side)
            caps = [cap_front, cap_side]
    else:
        if use_phone_streams:
            phone_url = "http://192.168.1.237:8081"
            client = IPWebcamClient(phone_url)
            client.start_stream()
            if not wait_for_first_frame(client):
                logger.warning("Nie otrzymano pierwszej klatki z telefonu w ciągu kilku sekund. "
                               "Strumień może być opóźniony, ale spróbuję dalej.")

            phone_clients = [client]
            caps = [None]
        else:
            source = 0 if use_camera else str(project_root / 'data' / 'videos' / 'try1' / 'jurek_1_bok.mp4')
            cap = cv2.VideoCapture(source)
            caps = [cap]

        rules_single = ShoulderPressRules(view_type=view_type)
        rules_list = [rules_single]
        window_names = ['Cyber Coach - Live Training']
        view_names = [view_type]

    detector = PoseDetector(complexity=2)
    calc = JointAngleCalculator(visibility_threshold=0.5)

    p_time = 0.0
    frame_idx = 0

    ANGLE_TO_IDX = {
        "left_elbow": 13, "right_elbow": 14,
        "left_knee": 25, "right_knee": 26,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24
    }

    color_ok = (0, 255, 0)
    color_error = (0, 0, 255)
    color_neutral = (200, 200, 200)

    last_rep_messages = [None] * len(caps)
    last_rep_times = [0.0] * len(caps)
    message_duration = 3.0

    confirmed_reps = 0

    # pomocnicze przechowywanie ostatnio wykrytych repów per widok (do synchronizacji dual view)
    recent_rep_by_view = {}

    print(f"Tryb: {'Oba widoki (synchronizacja)' if enable_dual_view else view_type}")
    print(f"Źródło: {'Kamery na żywo' if use_camera else 'Pliki wideo' if not use_phone_streams else 'Telefony (IP Webcam)'}")
    print("Naciśnij 'q' aby zakończyć\n")

    try:
        while True:
            frames = []
            all_ended = True

            # Pobieranie klatek - obsługa telefonów lub VideoCapture
            for idx in range(len(caps)):
                frame = None
                # jeśli mamy klientów telefonu i odpowiadający klient istnieje, pobierz z niego
                if phone_clients and idx < len(phone_clients):
                    try:
                        frame = phone_clients[idx].get_current_frame()
                    except Exception:
                        frame = None
                else:
                    cap = caps[idx]
                    if cap is not None:
                        try:
                            ret, f = cap.read()
                            if ret:
                                frame = f
                        except Exception:
                            frame = None

                frames.append(frame)
                if frame is not None:
                    all_ended = False

            if all_ended:
                print("Koniec wideo / brak klatek.")
                break

            # Przetwarzanie klatek
            for i, (frame, rule_set, window_name, view_name) in enumerate(
                    zip(frames, rules_list, window_names, view_names)):
                if frame is None:
                    # pokaż puste okno żeby UI nie znikło (opcjonalne)
                    black = 255 * 0 * frame if False else None
                    # pomijamy dalsze przetwarzanie
                    continue

                # Detekcja pozy i landmarków
                # handle voice-controlled pause/resume
                with detection_lock:
                    enabled = detection_enabled

                if not enabled:
                    # show a small overlay indicating detection is paused
                    h, w = frame.shape[:2]
                    cv2.putText(frame, "DETECTION PAUSED (voice)", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                    landmarks = None
                    angles = {}
                else:
                    frame = detector.find_pose(frame, draw=True)
                    landmarks = detector.get_landmarks()
                    h, w = frame.shape[:2]
                    channels = frame.shape[2] if len(frame.shape) == 3 else 1

                    angles = {}
                    if landmarks:
                        angles = calc.get_all_angles(landmarks, (h, w, channels))

                    # zapisujemy kąty do widoków
                    if view_name == 'front':
                        angles_front = angles
                    elif view_name == 'side':
                        angles_side = angles

                # show latest voice message briefly
                if last_voice_msg and (time.time() - last_voice_time) < voice_msg_duration:
                    cv2.putText(frame, last_voice_msg, (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                # feedback (błędy techniczne)
                has_errors = rule_set.has_angle_errors(angles) if enable_feedback else False

                # update detekcji powtórzeń
                completed_rep = rule_set.update_repetition_tracking(angles, frame_idx)

                if completed_rep:
                    # przygotuj komunikat dla użytkownika
                    status_msg = "OK" if completed_rep.is_complete else "NIEPOPRAWNE"
                    msg_color = color_ok if completed_rep.is_complete else color_error
                    rom = completed_rep.rom
                    last_rep_messages[i] = (status_msg, msg_color, rom)
                    last_rep_times[i] = time.time()

                    # synchronizacja przy dual view: sprawdź rep w drugim widoku
                    if enable_dual_view:
                        recent_rep_by_view[view_name] = (completed_rep, frame_idx, completed_rep.is_complete)
                        # sprawdź czy jest rep w drugim widoku w bliskim czasie
                        other_view = 'side' if view_name == 'front' else 'front'
                        other = recent_rep_by_view.get(other_view)
                        if other:
                            other_rep, other_frame_idx, other_ok = other
                            # prosta heurystyka synchronizacji: bliski czas wykrycia
                            if abs(other_rep.start_frame - completed_rep.start_frame) < SYNC_FRAME_THRESHOLD:
                                # jeśli oba widoki OK -> zatwierdzamy
                                if completed_rep.is_complete and other_ok:
                                    confirmed_reps += 1
                                    # ustaw komunikaty dla obu okien
                                    # znajdujemy indeksy okien i ustawiamy wiadomości
                                    for j, vn in enumerate(view_names):
                                        if vn in (view_name, other_view):
                                            last_rep_messages[j] = ("ZATWIERDZONE", color_ok, rom)
                                            last_rep_times[j] = time.time()
                            # usuń starsze repy, żeby nie mnożyć potwierdzeń
                            recent_rep_by_view.pop(other_view, None)
                            recent_rep_by_view.pop(view_name, None)
                        else:
                            # single view: jeśli is_complete to zwiększ licznik zatwierdzonych powtórzeń
                            if completed_rep.is_complete:
                                confirmed_reps += 1

                # Rysowanie kątów obok landmarków
                for joint_name, angle in angles.items():
                    if angle is None:
                        continue
                    if joint_name not in ANGLE_TO_IDX:
                        continue
                    idx_lm = ANGLE_TO_IDX[joint_name]
                    try:
                        lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
                        lm = lm_list[idx_lm]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        color = color_ok if not rule_set.has_angle_errors({joint_name: angle}) else color_error
                        cv2.putText(frame, f"{int(angle)}", (x + 15, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.circle(frame, (x, y), 6, color, -1)
                    except Exception:
                        continue

                # HUD / FPS / liczba powtórzeń
                c_time = time.time()
                fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
                p_time = c_time

                # czarny pasek i tekst
                cv2.rectangle(frame, (0, 0), (420, 120), (0, 0, 0), -1)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Powtorzenia (zatw.): {confirmed_reps}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ok, 2)

                if last_rep_messages[i] is not None:
                    if (time.time() - last_rep_times[i]) < message_duration:
                        status_msg, msg_color, rom = last_rep_messages[i]
                        if view_name == 'front':
                            cv2.putText(frame, f"{status_msg} | ROM: {rom:.1f} deg", (10, 105),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)

                cv2.imshow(window_name, frame)

            frame_idx += 1

            # obsługa klawisza q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nZakonczono przez użytkownika")
                break

    finally:
        # cleanup: zwolnij VideoCapture i zatrzymaj klientów telefonu
        for cap in caps:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        for client in (phone_clients or []):
            try:
                client.stop_stream()
            except Exception:
                pass

        # stop background voice listener
        try:
            stop_listening()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print("\n=== PODSUMOWANIE ===")
        for i, (rule_set, view_name) in enumerate(zip(rules_list, view_names)):
            summary = rule_set.get_repetition_summary()
            print(f"\n{view_name.upper()} VIEW:")
            print(f"  Wszystkie wykryte: {summary['total_reps']}")
            print(f"  Prawidłowe: {summary['complete_reps']}")
            print(f"  Nieprawidłowe: {summary['incomplete_reps']}")
            if summary['complete_reps'] > 0:
                print(f"  Średni ROM: {summary['avg_rom']:.1f} deg")

        if enable_dual_view:
            print(f"\nZATWIERDZONE (oba widoki OK): {confirmed_reps}")
        else:
            print(f"\nZATWIERDZONE: {confirmed_reps}")


if __name__ == '__main__':
    main()