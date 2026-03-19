from typing_extensions import Generator, List, Union
import numpy as np
import torch
from pathlib import Path
from mmdet3d.structures.bbox_3d import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes
import tqdm
import pickle

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
    
    K_inv_left = np.linalg.inv(P2[:, :3])
    
    for i, (left_box, right_box) in enumerate(zip(bboxes_2d_left, bboxes_2d_right)):
        
        left_box_aug = left_box.copy()
        right_box_aug = right_box.copy()
        
        n_aug = num_augmentaions[labels[i]]
        
        for _ in range(n_aug):
            if train:
                left_box_aug = random_shift_enlarge_box2d(left_box, shift_ratio=0.05, enlarge_ratio=0.0)
            # else:
            #     # only enlarge
            #     left_box_aug = random_shift_enlarge_box2d(left_box, 0, 0.1)
            #     right_box_aug = random_shift_enlarge_box2d(right_box, 0, 0.1)
            
            # intersection of the fields of view of the two images
            fov_inds = (coord_image_left[:, 0] <= left_box_aug[2]) & \
                (coord_image_left[:, 0] >= left_box_aug[0]) & \
                (coord_image_left[:, 1] <= left_box_aug[3]) & \
                (coord_image_left[:, 1] >= left_box_aug[1])
                
            points_fov = points[fov_inds, :]
            
            left_center = (left_box_aug[2:4] + left_box_aug[0:2]) / 2
            right_center = (right_box_aug[2:4] + right_box_aug[0:2]) / 2
            left_dims = left_box[2:4] - left_box[0:2]
            right_dims = right_box[2:4] - right_box[0:2]
            
            left_center_hom = np.concatenate([left_center, [1]])[:, np.newaxis]
            right_center_hom = np.concatenate([right_center, [1]])[:, np.newaxis]
            
            # backprojection = point at the infinity -> append 0 as last coordinate
            backprojections = np.append(K_inv_left.dot(left_center_hom).flatten(), [0])[np.newaxis, :]
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
            likelihoods_left = np.exp(-np.sum((left_coors[:, 0:2] - left_center)**2 / (2 * left_dims**2), axis=1))            
            # centered_points = np.concatenate([centered_points, likelihoods_left[:, np.newaxis], 
            #                                   likelihoods_right[:, np.newaxis]], axis=1)
            centered_points = np.concatenate([centered_points, likelihoods_left[:, np.newaxis]], axis=1)
            
            yield i, fov_inds, centered_points, yaw_lidar, rotation_matrix
        

def make_dataset(velo_dir: Path, calib_dir: Path, gt_dir: Path, ids_path: Path, 
                 min_points_per_frustum: int, out_path: Path, classes: List[str],
                 labels_mapping: dict, point_cloud_range: List[int], 
                 num_augmentaions_per_class: Union[dict, int] = 1, 
                 train: bool = True):
    
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
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
        
        if bboxes_3d_kitti.shape[0] == 0:
            # in case there are no objects of the classes of interest
            continue
        
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
            
            if rotated_frustum_points.shape[0] > min_points_per_frustum:  #and rotated_frustum_points.shape[0] < 512: # and truncated[object_id] == 0:
                # rt_matrix_4x4 = np.zeros((4, 4), dtype=np.float32)
                # rt_matrix_4x4[:3, :3] = rt_matrix
                # rt_matrix_4x4[3, 3] = 1
                # lidar_to_cam_sample = lidar_to_cam @ np.linalg.inv(rt_matrix_4x4)
                # cam_to_lidar_sample = np.linalg.inv(lidar_to_cam_sample)
                
                bbox = bboxes_3d_lidar[object_id, :].copy()[np.newaxis, :]
                bbox[:, :3] = bbox[:, :3] @ rt_matrix.T
                bbox[:, -1] -= frustum_yaw
            
                # bbox = bboxes_3d_cam[object_id, :].copy()
                # bbox_center = np.append(bbox[:3], [1])[np.newaxis, :] @ cam_to_lidar.T
                # bbox_center[:, :3] /= bbox_center[:, 3:]
                # bbox_center[:, 3:] = 1
                # bbox_center[:, :3] = bbox_center[:, :3] @ rt_matrix.T
                # # bbox_center[:, :2] -= bev_centroid
                # bbox_center = bbox_center @ lidar_to_cam.T
                # bbox_center[:, :3] /= bbox_center[:, 3:]
                # bbox[:3] = bbox_center[0, :3]
                # bbox[-1] -= frustum_yaw
                # bbox = bbox[np.newaxis, :]
                # bbox = CameraInstance3DBoxes(bbox).convert_to(Box3DMode.LIDAR, rt_mat=cam_to_lidar).tensor.numpy()
                
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
                
                
if __name__ == '__main__': 
    
    VALID_CLASSES = ['Pedestrian', 'Cyclist', 'Car']
    labels_mapping = {VALID_CLASSES[i]: i for i in range(len(VALID_CLASSES))}
    
    # make_dataset(
    #     velo_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/velodyne"),
    #     calib_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/calib"),
    #     gt_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/label_2"),
    #     ids_path=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/ImageSets/train.txt"),
    #     min_points_per_frustum=10,
    #     out_path=Path("/mnt/proj2/dd-24-8/frustum_datasets/v1_single_view_no_align/kitti_frustum_info_train.pkl"),
    #     classes=VALID_CLASSES,
    #     labels_mapping=labels_mapping,
    #     point_cloud_range=[0, -40, -3, 100, 40, 1],
    #     num_augmentaions_per_class={'Car': 10, 'Pedestrian': 10, 'Cyclist': 10},
    #     train=True,
    # )
    
    # make_dataset(
    #     velo_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/velodyne"),
    #     calib_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/calib"),
    #     gt_dir=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/training/label_2"),
    #     ids_path=Path("/mnt/proj2/dd-24-8/kitti_mmdet3d/ImageSets/val.txt"),
    #     min_points_per_frustum=10,
    #     out_path=Path("/mnt/proj2/dd-24-8/frustum_datasets/v1_single_view_no_align/kitti_frustum_info_val.pkl"),
    #     classes=VALID_CLASSES,
    #     labels_mapping=labels_mapping,
    #     point_cloud_range=[0, -40, -3, 100, 40, 1],
    #     num_augmentaions_per_class=1,
    #     train=False
    # )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/train.txt"),
        min_points_per_frustum=10,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_car_info_train.pkl"),
        classes=['Car'],
        labels_mapping={'Car': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class={'Car': 10},
        train=True,
    )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/val.txt"),
        min_points_per_frustum=10,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_car_info_val.pkl"),
        classes=['Car'],
        labels_mapping={'Car': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class=1,
        train=False
    )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/train.txt"),
        min_points_per_frustum=5,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_pedestrian_info_train.pkl"),
        classes=['Pedestrian'],
        labels_mapping={'Pedestrian': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class={'Pedestrian': 15},
        train=True,
    )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/val.txt"),
        min_points_per_frustum=5,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_pedestrian_info_val.pkl"),
        classes=['Pedestrian'],
        labels_mapping={'Pedestrian': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class=1,
        train=False
    )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/train.txt"),
        min_points_per_frustum=5,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_cyclist_info_train.pkl"),
        classes=['Cyclist'],
        labels_mapping={'Cyclist': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class={'Cyclist': 20},
        train=True,
    )
    
    make_dataset(
        velo_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/velodyne"),
        calib_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/calib"),
        gt_dir=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/training/label_2"),
        ids_path=Path("/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/val.txt"),
        min_points_per_frustum=5,
        out_path=Path("/mnt/datasets_1/carlos00/frustum_datasets/kitti_single_class/kitti_frustum_cyclist_info_val.pkl"),
        classes=['Cyclist'],
        labels_mapping={'Cyclist': 0},
        point_cloud_range=[0, -40, -3, 100, 40, 1],
        num_augmentaions_per_class=1,
        train=False
    )