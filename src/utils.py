import scipy
from scipy.linalg import null_space
import numpy as np
from pathlib import Path
import torch
from typing_extensions import List


def extract_rotation_translation(P: np.ndarray):
    fx = P[0, 0]
    fy = P[1, 1]
    f = (fx + fy) / 2

    R = P[:, :3] / f
    T = P[:, 3] / f

    return R, T


def get_camera_center(P: np.ndarray):
    camera_center = null_space(P).flatten()
    camera_center /= camera_center[-1]
    return camera_center


def decompose_projection_matrix(P):
    K, R = scipy.linalg.rq(P[:, :3])
    K /= K[2, 2]
    
    t = np.linalg.inv(K) @ P[:, 3:]
    
    return K, R, t.flatten()


def compute_fundamental_matrix(P2, P3):
    K2, R2, t2 = decompose_projection_matrix(P2)
    K3, R3, t3 = decompose_projection_matrix(P3)

    R = R3 @ R2.T
    t = t3 - R @ t2

    t_x = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

    E = t_x @ R
    F = np.linalg.inv(K3).T @ E @ np.linalg.inv(K2)
    return F


def read_kitti_calibration_data(calib_file):
    calib_data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            try:
                key, values = line.strip().split(':')
            except ValueError:
                # If the line doesn't contain values it is ignored
                continue
            values = np.array([float(x) for x in values.split()]).reshape((3, -1))  # R0_rect is 3×3, others 3×4

            if key.startswith('Tr'):
                # Adds the row for homogeneous coordinates to calibration matrices 
                values = np.concatenate([values, [[0, 0, 0, 1]]], axis=0)
            calib_data[key] = values
    
    if 'R0_rect' in calib_data:
        R0_rect = calib_data['R0_rect']
        R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
        R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
        calib_data['R0_rect'] = R0_rect
    return calib_data


def read_kitti_point_cloud(bin_path: Path, point_cloud_range: List[float]):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1,4))
    for i in range(3):
        scan = scan[(scan[:, i] > point_cloud_range[i]) & (scan[:, i] < point_cloud_range[i+3])]
    return scan


def read_kitti_labels(labels_path: Path, keep_dont_care: bool = False, classes: List[str] = None):
    with open(labels_path) as labels_file:
        labels_data = labels_file.readlines()

    bboxes = []
    bboxes_3d = []
    labels = []
    truncation = []
    occlusion = []
    alpha = []
    for label in labels_data:
        label = label.split()
        if label[0] in classes:
            truncation.append(label[1].rstrip('\n'))
            occlusion.append(label[2].rstrip('\n'))
            alpha.append(label[3].rstrip('\n'))
            labels.append(label[0])
            bboxes.append([float(label[4]), float(label[5]), float(label[6]), float(label[7])])
            bboxes_3d.append([float(x) for x in label[8:15]])
    bboxes = np.array(bboxes).astype(np.float32)
    bboxes_3d = np.array(bboxes_3d).astype(np.float32)
    return truncation, occlusion, alpha, bboxes_3d, bboxes, labels


def calibration_to_torch(calibration_data, device='cuda:0'):
    calibration_data['P2'] = torch.tensor(calibration_data['P2'], dtype=torch.float32, device=device)
    calibration_data['P3'] = torch.tensor(calibration_data['P3'], dtype=torch.float32, device=device)
    calibration_data['Tr_velo_to_cam'] = torch.tensor(calibration_data['Tr_velo_to_cam'], dtype=torch.float32, device=device) 
    if 'R0_rect' in calibration_data:
        calibration_data['R0_rect'] = torch.tensor(calibration_data['R0_rect'], dtype=torch.float32, device=device)
    if 'F' in calibration_data:
        calibration_data['F'] = torch.tensor(calibration_data['F'], dtype=torch.float32, device=device)
    if 'PI' in calibration_data:
        calibration_data['PI'] = torch.tensor(calibration_data['PI'], dtype=torch.float32, device=device)
    return calibration_data 