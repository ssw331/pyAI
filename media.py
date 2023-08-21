# 微软我***你*
import mediapipe as mp
import cv2
import time
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFont, ImageDraw


# 汇总所有点的XYZ坐标
def get_x(each):
    return each.x


def get_y(each):
    return each.y


def get_z(each):
    return each.z


NOSE = 0  # 鼻
LEYE = 2  # 左眼
REYE = 5  # 右眼
LEAR = 7  # 左耳
REAR = 8  # 右耳
LSD = 11  # 左肩
RSD = 12  # 右肩
LEl = 13  # 左臂
REl = 14  # 右臂
LWR = 15  # 左手腕
RWR = 16  # 右手腕
LBL = 23  # 左大腿
RBL = 24  # 右大腿
LSL = 25  # 左小腿
RSL = 26  # 右小腿
LF = 27  # 左脚
RF = 28  # 右脚


# p0为基准点
def getAngle(p1, p2, p0):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p0 = np.array(p0)

    v1 = p1 - p0
    v2 = p2 - p0

    angle = np.dot(v1, v2) / (np.sqrt(np.sum(np.dot(v1, v1))) * np.sqrt(np.sum(np.dot(v2, v2))))
    angle = np.arccos(angle) / 3.1415 * 180

    if angle < 0:
        angle *= -1

    return angle


def eye_parallel(keypoints):
    look_up = 15
    look_down = -5
    eye2ear = keypoints[REAR][1] - keypoints[REYE][1]
    if look_down < eye2ear < look_up:
        print('eye pass\n')
    else:
        print('eye fail\n')


def arm_straight(keypoints):
    arm_max = 180
    arm_min = 135
    ang = getAngle(keypoints[RSD], keypoints[RWR], keypoints[REl])
    print('手臂弯曲度：', ang)
    if arm_min < ang <= arm_max:
        print('arm pass\n')
    else:
        print('arm fail\n')


def poseIdentify(keypoints):
    pose_result_name = ""
    keypoints = np.array(keypoints)

    # 以观察者为主视角区分左右
    angle_left_arm = getAngle(keypoints[15], keypoints[11], keypoints[13])
    angle_right_arm = getAngle(keypoints[12], keypoints[16], keypoints[14])
    angle_left_leg = getAngle(keypoints[27], keypoints[23], keypoints[25])
    angle_right_leg = getAngle(keypoints[28], keypoints[24], keypoints[26])

    # 动作区分，需要角度

    return pose_result_name


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

        coords = np.array(results.pose_landmarks.landmark)
        # 分别获取关键点XYZ坐标
        points_x = np.array(list(map(get_x, coords)))
        points_y = np.array(list(map(get_y, coords)))
        points_z = np.array(list(map(get_z, coords)))
        # 将三个方向坐标合并
        points = np.vstack((points_x, points_y, points_z)).T

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
            .get_default_pose_landmarks_style()
        )

        eye_parallel(points)
        arm_straight(points)
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
