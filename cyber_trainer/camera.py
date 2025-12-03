from posedetector import PoseDetector
import cv2
import time

def main():

    source = 0 # source config
    window_name = 'Cyber Coach - Camera Test Preview'

    cap = cv2.VideoCapture(source)
    detector = PoseDetector(complexity=2)
    p_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error or end of video.")
            break

        frame = detector.find_pose(frame, draw=True)

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