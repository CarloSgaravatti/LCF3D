from typing_extensions import Generator, List, Union
import numpy as np
import torch
from pathlib import Path
from mmdet3d.structures.bbox_3d import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
import tqdm
import pickle
import argparse

# append in sys the parent directory to be able to import utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    read_kitti_calibration_data, 
    read_kitti_point_cloud, 
    read_kitti_labels, 
    calibration_to_torch,
    compute_fundamental_matrix,
)
from bbox_utils import project_bboxes


def random_shift_enlarge_box2d(box2d, shift_ratio=0.1, enlarge_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    er = enlarge_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r+er) # 0.9+er to 1.1+er
    w2 = w*(1+np.random.random()*2*r-r+er) # 0.9+er to 1.1+er
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])


def get_bbox_fov_stereo(bboxes_2d_left: np.ndarray, bboxes_2d_right: np.ndarray, 
                        points: np.ndarray, P2: np.ndarray, P3: np.ndarray, 
                        R0_rect: np.ndarray, Tr_velo_to_cam: np.ndarray,
                        num_augmentaions: dict, train: bool, labels: List) -> Generator:
    
    camera_coordinates = R0_rect.dot(Tr_velo_to_cam.dot(np.insert(points[:, :3], 3, 1, axis=1).T))
    
    coord_image_left = P2.dot(camera_coordinates)
    coord_image_left[:2] /= coord_image_left[2,:]
    coord_image_left[2] = 1
    coord_image_left = coord_image_left.T

    coord_image_right = P3.dot(camera_coordinates)
    coord_image_right[:2] /= coord_image_right[2,:]
    coord_image_right[2] = 1
    coord_image_right = coord_image_right.T
    
    K_inv_left = np.linalg.inv(P2[:, :3])
    K_inv_right = np.linalg.inv(P3[:, :3])
    
    for i, (left_box, right_box) in enumerate(zip(bboxes_2d_left, bboxes_2d_right)):
        
        left_box_aug = left_box.copy()
        right_box_aug = right_box.copy()
        
        n_aug = num_augmentaions[labels[i]]
        
        for _ in range(n_aug):
            if train:
                left_box_aug = random_shift_enlarge_box2d(left_box, shift_ratio=0.05, enlarge_ratio=0.0)
                right_box_aug = random_shift_enlarge_box2d(right_box, shift_ratio=0.05, enlarge_ratio=0.0)
            
            # intersection of the fields of view of the two images
            fov_inds = (coord_image_left[:, 0] <= left_box_aug[2]) & \
                (coord_image_left[:, 0] >= left_box_aug[0]) & \
                (coord_image_left[:, 1] <= left_box_aug[3]) & \
                (coord_image_left[:, 1] >= left_box_aug[1]) & \
                (coord_image_right[:, 0] <= right_box_aug[2]) & \
                (coord_image_right[:, 0] >= right_box_aug[0]) & \
                (coord_image_right[:, 1] <= right_box_aug[3]) & \
                (coord_image_right[:, 1] >= right_box_aug[1])
                
            points_fov = points[fov_inds, :]
            
            left_center = (left_box_aug[2:4] + left_box_aug[0:2]) / 2
            right_center = (right_box_aug[2:4] + right_box_aug[0:2]) / 2
            left_dims = left_box[2:4] - left_box[0:2]
            right_dims = right_box[2:4] - right_box[0:2]
            
            left_center_hom = np.concatenate([left_center, [1]])[:, np.newaxis]
            right_center_hom = np.concatenate([right_center, [1]])[:, np.newaxis]
            
            # backprojection = point at the infinity -> append 0 as last coordinate
            left_backproj = np.append(K_inv_left.dot(left_center_hom).flatten(), [0])[np.newaxis, :]
            right_backproj = np.append(K_inv_right.dot(right_center_hom).flatten(), [0])[np.newaxis, :]
            
            backprojections = np.vstack([left_backproj, right_backproj])
            backprojections = np.linalg.inv(R0_rect @ Tr_velo_to_cam).dot(backprojections.T).T
            
            yaw_lidar = np.arctan2(backprojections[:, 1], backprojections[:, 0])[0]
            
            # rotating by the opposite of the frustum orientation in order to have it parallel to forward axis
            rotation_matrix = np.array([
                [np.cos(-yaw_lidar), -np.sin(-yaw_lidar), 0],
                [np.sin(-yaw_lidar), np.cos(-yaw_lidar), 0],
                [0, 0, 1],
            ])
            centered_points = points_fov.copy()
            centered_points[:, :3] = centered_points[:, :3] @ rotation_matrix.T
            
            left_coors = coord_image_left[fov_inds]
            right_coors = coord_image_right[fov_inds]
            likelihoods_left = np.exp(-np.sum((left_coors[:, 0:2] - left_center)**2 / (2 * left_dims**2), axis=1))
            likelihoods_right = np.exp(-np.sum((right_coors[:, 0:2] - right_center)**2 / (2 * right_dims**2), axis=1))
            
            max_likelihood = np.maximum(likelihoods_left, likelihoods_right)
            
            centered_points = np.concatenate([centered_points, max_likelihood[:, np.newaxis]], axis=1)
            
            yield i, fov_inds, centered_points, yaw_lidar, rotation_matrix
        

def make_dataset(velo_dir: Path, calib_dir: Path, gt_dir: Path, ids_path: Path, 
                 min_points_per_frustum: int, out_path: Path, classes: List[str],
                 labels_mapping: dict, point_cloud_range: List[int], 
                 num_augmentaions_per_class: Union[dict, int] = 1, 
                 train: bool = True):
    
    if type(num_augmentaions_per_class) == int:
        num_augmentaions_per_class = {class_: num_augmentaions_per_class for class_ in classes}
    
    with open(ids_path, 'r') as split_ids_file:
        sample_ids = split_ids_file.readlines()
    sample_ids = [sample_id.rstrip('\n') for sample_id in sample_ids]
    
    data_list = []
    for sample_id in tqdm.tqdm(sample_ids):
        velo_path = velo_dir / f'{sample_id}.bin'
        labels_path = gt_dir / f'{sample_id}.txt'
        calib_path = calib_dir / f'{sample_id}.txt'
        
        calibration_data = read_kitti_calibration_data(calib_path)
        calibration_data['F'] = compute_fundamental_matrix(calibration_data['P2'], calibration_data['P3'])
        calibration_data = calibration_to_torch(calibration_data, device='cpu')
        point_cloud = read_kitti_point_cloud(velo_path, point_cloud_range)
        truncated, occluded, _, bboxes_3d_kitti, bboxes_left, labels = read_kitti_labels(labels_path, keep_dont_care=False, classes=classes)
        
        bboxes_3d_cam = bboxes_3d_kitti[:, (3, 4, 5, 2, 0, 1, 6)].copy()
        bboxes_3d_project = bboxes_3d_cam.copy()
        bboxes_3d_project[:, 1] -= bboxes_3d_project[:, 4] / 2
        bboxes_right = project_bboxes(torch.tensor(bboxes_3d_project, dtype=torch.float32), calibration_data['P3'],
                                      lidar=False, R_rect=calibration_data['R0_rect']).numpy()
        bboxes_3d_lidar = Box3DMode.convert(bboxes_3d_cam.copy(), src=Box3DMode.CAM, dst=Box3DMode.LIDAR,
                                            rt_mat=np.linalg.inv(calibration_data['R0_rect'].numpy() @ calibration_data['Tr_velo_to_cam'].numpy()))
        bboxes_3d_lidar = LiDARInstance3DBoxes(bboxes_3d_lidar)
        segmentation_masks = bboxes_3d_lidar.points_in_boxes_all(
            torch.tensor(point_cloud[:, :3], dtype=torch.float32, device='cuda:0')).cpu().numpy()
        bboxes_3d_lidar = bboxes_3d_lidar.tensor.numpy()
        
        data_iterator = get_bbox_fov_stereo(bboxes_2d_left=bboxes_left, bboxes_2d_right=bboxes_right, points=point_cloud,
                                            P2=calibration_data['P2'].numpy(), P3=calibration_data['P3'].numpy(), 
                                            R0_rect=calibration_data['R0_rect'].numpy(),
                                            Tr_velo_to_cam=calibration_data['Tr_velo_to_cam'].numpy(),
                                            num_augmentaions=num_augmentaions_per_class, train=train, labels=labels)
        
        lidar_to_cam = calibration_data['R0_rect'].numpy() @ calibration_data['Tr_velo_to_cam'].numpy()
        cam_to_lidar = np.linalg.inv(lidar_to_cam)
        
        i = 0
        for object_id, fov_inds, rotated_frustum_points, frustum_yaw, rt_matrix in data_iterator:
            
            if rotated_frustum_points.shape[0] > min_points_per_frustum:
                
                bbox = bboxes_3d_lidar[object_id, :].copy()[np.newaxis, :]
                bbox[:, :3] = bbox[:, :3] @ rt_matrix.T
                bbox[:, -1] -= frustum_yaw
                
                one_hot_vector = np.zeros(len(classes))
                one_hot_vector[labels_mapping[labels[object_id]]] = 1
                
                sample_dict = {
                    'ori_id': sample_id,
                    'object_id': sample_id + f'{object_id:03d}',
                    'inner_sample_id': sample_id + f'{i:03d}',
                    'points': rotated_frustum_points.copy(),
                    'pts_semantic_mask': segmentation_masks[fov_inds, object_id].copy(),
                    'lidar_to_cam': lidar_to_cam, #lidar_to_cam_sample,
                    'cam_to_lidar': cam_to_lidar, #cam_to_lidar_sample,
                    'cam_to_img': calibration_data['P2'].numpy(),
                    'frustum_angle': frustum_yaw,
                    'gt_bboxes_left': [bboxes_left[object_id]],
                    'gt_bboxes_right': [bboxes_right[object_id]],
                    'gt_labels': [labels_mapping[labels[object_id]]],
                    'gt_bboxes_3d': bbox,
                    'gt_labels_3d': [labels_mapping[labels[object_id]]],
                    'one_hot_vector': one_hot_vector,
                }
                data_list.append(sample_dict)
                
            i += 1
            
    infos = {
        'metainfo': {
            'dataset_type': 'frustum_dataset',
            'task_name': 'localization',
        },
        'data_list': data_list
    }
    with open(out_path, 'wb') as fp:
        pickle.dump(infos, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='Create frustum dataset from KITTI')
    parser.add_argument('--kitti_path', type=str, required=True, help='Path to the kitti dataset')
    parser.add_argument('--min_points_per_frustum', type=int, default=10, help='Minimum points per frustum')
    parser.add_argument('--out_path', type=Path, required=True, help='Output path for the dataset')
    return parser.parse_args()
                
                
if __name__ == '__main__': 
    args = parse_args()
    
    VALID_CLASSES = ['Pedestrian', 'Cyclist', 'Car']
    labels_mapping = {VALID_CLASSES[i]: i for i in range(len(VALID_CLASSES))}

    kitti_path = Path(args.kitti_path)
    output_path = Path(args.out_path)
    output_path.mkdir(parents=True, exist_ok=True)
    min_points_per_frustum = args.min_points_per_frustum
    
    make_dataset(
        velo_dir=kitti_path / "training/velodyne",
        calib_dir=kitti_path / "training/calib",
        gt_dir=kitti_path / "training/label_2",
        ids_path=kitti_path / "ImageSets/train.txt",
        min_points_per_frustum=min_points_per_frustum,
        out_path=output_path / "kitti_frustum_info_train.pkl",
        classes=VALID_CLASSES,
        labels_mapping=labels_mapping,
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class={'Car': 10, 'Pedestrian': 10, 'Cyclist': 10},
        train=True,
    )
    
    make_dataset(
        velo_dir=kitti_path / "training/velodyne",
        calib_dir=kitti_path / "training/calib",
        gt_dir=kitti_path / "training/label_2",
        ids_path=kitti_path / "ImageSets/val.txt",
        min_points_per_frustum=min_points_per_frustum,
        out_path=output_path / "kitti_frustum_info_val.pkl",
        classes=VALID_CLASSES,
        labels_mapping=labels_mapping,
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class=1,
        train=False
    )