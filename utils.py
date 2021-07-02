import cv2
import numpy as np
import math


def calculate_angles(landmarks_2D, cam_w=256, cam_h=256):
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    indices_for_projection = [17, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # лява вежда
        [-6.825897, 6.760612, 4.402142],  # дясна вежда
        [5.311432, 5.485328, 3.987654],  # ляво око - ляв край
        [1.789930, 5.393625, 4.413414],  # ляво око - десен край
        [-1.789930, 5.393625, 4.413414],  # дясно око - ляв край
        [-5.311432, 5.485328, 3.987654],  # дясно око - десен край
        [-2.005628, 1.409845, 6.165652],  # лява ноздра
        [-2.005628, 1.409845, 6.165652],  # дясна ноздра
        [2.774015, -2.080775, 5.048531],  # уста - ляв край
        [-2.774015, -2.080775, 5.048531],  # уста - десен край
        [0.000000, -3.116408, 6.097667],  # долна устна
        [0.000000, -7.415691, 4.070434],  # брада
    ])
    euler_angles_landmark = []
    for index in indices_for_projection:
        euler_angles_landmark.append(landmarks_2D[index])
    euler_angles_landmark = np.asarray(euler_angles_landmark, dtype=np.float32)

    _, rvec, tvec = cv2.solvePnP(landmarks_3D, euler_angles_landmark, camera_matrix, camera_distortion)
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return np.fromiter(map(lambda k: k[0] * np.pi / 180.0, euler_angles), dtype=np.float32)  # pitch, yaw, roll

def custom_sample_wights(euler_angle_gt, landmarks):
    angles = calculate_angles(landmarks.reshape(-1, 2))
    weight_angle = np.sum(1 - np.cos(angles - euler_angle_gt))
    
    return weight_angle

def custom_nme(landmark_gt, landmarks):
    normalization_coeff = np.linalg.norm(landmark_gt[45] - landmark_gt[36])
    return np.mean([np.linalg.norm(x) for x in (landmark_gt - landmarks)])