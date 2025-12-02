import cv2
import mediapipe as mp


# mediapipe pose init
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
connection_style = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)


# source config
s = 0 # laptop camera (temp)
wn = 'Camera Test Preview' # same as above
cap = cv2.VideoCapture(s)


# main loop
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break


    result = pose.process(frame)
    landmarks = result.pose_landmarks

    # mediapipe returns a list of landmarks with x, y, z, visibility
    # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=pl
    # if landmarks:
    #     for id, lm in enumerate(landmarks.landmark):
    #         print(id, lm.x, lm.y, lm.z, lm.visibility)

    # drawing landmarks
    if landmarks:

        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=connection_style
        )


    cv2.imshow(wn, frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow(wn)