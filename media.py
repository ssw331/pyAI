# 微软我***你*
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture("test.mp4")
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # 加载一个视频的话，把continue换成break
            # 这个是注释
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR图转RGB
        results = holistic.process(image)  # 处理三通道彩色图
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB转BGR
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.face_landmarks,  # 画脸
        #     mp_holistic.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,  # 画脸
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                print(index, landmarks)
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
