import cv2
import mediapipe as mp


# mediapipe pose init
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# source config
s = 0 # laptop camera (temp)
wn = 'Camera Test Preview' # same as above


cap = cv2.VideoCapture(s)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    result = pose.process(frame)
    landmarks = result.pose_landmarks

    if landmarks:

        if landmarks:
            for id, lm in enumerate(landmarks.landmark):
                print(id, lm.x, lm.y, lm.z, lm.visibility)



    cv2.imshow(wn, frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow(wn)