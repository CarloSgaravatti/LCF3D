import torch
from torch import Tensor
import numpy as np
from mmdet.models.task_modules import BaseAssigner, AssignResult, BboxOverlaps2D
from mmengine.structures import InstanceData
from utils import get_camera_center
from scipy.optimize import linear_sum_assignment
from scipy.linalg import null_space

ANCHOR_SIZES = [[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]]


def extract_corners(bboxes_3d: torch.Tensor, lidar: bool = False):
    centers = bboxes_3d[:, :3]
    half_dimensions = bboxes_3d[:, 3:6].reshape(-1, 1, 3).repeat(1, 8, 1) / 2
    mult = [-1, 1]
    corners_multipliers = bboxes_3d.new_tensor([[x_mult, y_mult, z_mult] for x_mult in mult for y_mult in mult for z_mult in mult])
    corners_relative = corners_multipliers * half_dimensions
    
    # directions for the yaw rotation
    direction_1 = torch.cos(bboxes_3d[:, -1])
    direction_2 = torch.sin(bboxes_3d[:, -1])
    
    # yaw rotation matrix to compute the rotated corners
    rotation_matrices = torch.zeros((bboxes_3d.shape[0], 3, 3), dtype=torch.float32, device=bboxes_3d.device)
    if lidar:
        rotation_matrices[:, 0, 0] = direction_1.view(-1)
        rotation_matrices[:, 0, 1] = -direction_2.view(-1)
        rotation_matrices[:, 1, 0] = direction_2.view(-1)
        rotation_matrices[:, 1, 1] = direction_1.view(-1)
        rotation_matrices[:, 2, 2] = 1
    else:
        rotation_matrices[:, 0, 0] = direction_1.view(-1)
        rotation_matrices[:, 0, 2] = -direction_2.view(-1)
        rotation_matrices[:, 2, 0] = direction_2.view(-1)
        rotation_matrices[:, 2, 2] = direction_1.view(-1)
        rotation_matrices[:, 1, 1] = 1
        
    # Reshaping corners to (N, 8, 3, 1) for broadcasting
    rotated_corners = torch.bmm(rotation_matrices, corners_relative.permute(0, 2, 1)).permute(0, 2, 1)
    rotated_corners = torch.squeeze(rotated_corners)
    
    # final corners in lidar/camera coordinates
    return rotated_corners + torch.unsqueeze(centers, 1)


def corners_to_img_coord(corners: torch.Tensor, P: torch.Tensor, lidar: bool = False, T: torch.Tensor = None):    
    
    # from lidar coordinates to camera coordinates
    corners_cam = torch.cat([corners, torch.ones_like(corners[:, :, :1])], dim=-1)
    num_boxes = corners_cam.shape[0]
    if T is not None and lidar:
        transformation_matrix = T.unsqueeze(0).expand(num_boxes, -1, -1)
        corners_cam = torch.matmul(corners_cam, transformation_matrix.transpose(1, 2))
    corners_cam /= corners_cam[:, :, 3:]

    # from camera coordinates to image coordinates
    projection_matrix = P.unsqueeze(0).expand(num_boxes, -1, -1)
    corners_projected = torch.matmul(corners_cam, projection_matrix.transpose(1, 2))
    corners_projected[:, :, :2] /= corners_projected[:, :, 2:]
    return corners_cam, corners_projected[:, :, :2]


def corners_to_axis_aligned_bbox(corners: torch.Tensor, P: torch.Tensor, lidar: bool = False, T: torch.Tensor = None):        
    corners_cam, corners_projected = corners_to_img_coord(corners, P, lidar, T)
    
    # extracting image bounding boxes with format xyxy
    min_x = torch.min(corners_projected[:, :, 0], dim=1)[0]
    min_y = torch.min(corners_projected[:, :, 1], dim=1)[0]
    max_x = torch.max(corners_projected[:, :, 0], dim=1)[0]
    max_y = torch.max(corners_projected[:, :, 1], dim=1)[0]
    
    return torch.stack((min_x, min_y, max_x, max_y), dim=1)


def axis_aligned_bboxes(corners_projected: torch.Tensor):
    min_x = torch.min(corners_projected[:, :, 0], dim=1)[0]
    min_y = torch.min(corners_projected[:, :, 1], dim=1)[0]
    max_x = torch.max(corners_projected[:, :, 0], dim=1)[0]
    max_y = torch.max(corners_projected[:, :, 1], dim=1)[0]
    
    return torch.stack((min_x, min_y, max_x, max_y), dim=1)


def project_corners(corners: torch.Tensor, P: torch.Tensor, lidar: bool = False, T: torch.Tensor = None):   
    corners_cam, corners_proj = corners_to_img_coord(corners, P, lidar, T)
    return axis_aligned_bboxes(corners_proj)
    

def project_bboxes(bboxes_3d: torch.Tensor, P: torch.Tensor, lidar: bool = False, T: torch.Tensor = None):    
    corners = extract_corners(bboxes_3d, lidar=lidar)
    return corners_to_axis_aligned_bbox(corners, P, lidar, T)


def match_bboxes_iou(bboxes_lidar: torch.Tensor, bboxes_image: torch.Tensor, iou_thr: float = 0.5, mode: str = 'iou'):
    if bboxes_lidar.shape[0] == 0:
        return torch.empty(size=(0,), dtype=torch.long)
    if bboxes_image.shape[0] == 0:
        return torch.zeros(size=(bboxes_lidar.shape[0],), dtype=torch.long)
    iou_calculator = BboxOverlaps2D()
    
    iou_matrix = iou_calculator(bboxes_lidar, bboxes_image, mode=mode)
    max_ious, max_indices = torch.max(iou_matrix, dim=1)
    matching = torch.where(max_ious >= iou_thr, max_indices + 1, 0)
    return matching


def match_bboxes_linear_sum_assign(iou_calculator, bboxes_lidar: torch.Tensor, bboxes_image: torch.Tensor,
                                   iou_thr: float = 0.5, mode: str = 'iou', conf_lambda: float = None,
                                   scores_lidar: torch.Tensor = None, scores_image: torch.Tensor = None):
    if bboxes_lidar.shape[0] == 0:
        return torch.empty(size=(0,), dtype=torch.int, device=bboxes_lidar.device)
    if bboxes_image.shape[0] == 0:
        return bboxes_lidar.new_zeros(size=(bboxes_lidar.shape[0],), dtype=torch.int)
    
    iou_matrix = iou_calculator(bboxes_lidar, bboxes_image, mode=mode)
    
    if conf_lambda is not None:
        confidence_weights = scores_lidar.cpu().unsqueeze(1).repeat(1, bboxes_image.shape[0])
        objective = -((1 - conf_lambda) * iou_matrix.cpu() - conf_lambda * confidence_weights)
    else:
        objective = -iou_matrix.cpu()
    
    lidar_ids, rgb_ids = linear_sum_assignment(objective)
    lidar_ids = bboxes_lidar.new_tensor(lidar_ids, dtype=torch.int)
    rgb_ids = bboxes_lidar.new_tensor(rgb_ids, dtype=torch.int)
    assigned_ious = iou_matrix[lidar_ids, rgb_ids]
    valid_matching = assigned_ious > iou_thr
    lidar_ids = lidar_ids[valid_matching]
    rgb_ids = rgb_ids[valid_matching]
    matching = bboxes_lidar.new_zeros(size=(bboxes_lidar.shape[0],), dtype=torch.int)
    matching[lidar_ids] = rgb_ids + 1
    return matching

def match_bboxes_linear_sum_assign_cls(bboxes_lidar: torch.Tensor, bboxes_image: torch.Tensor, 
                                       labels_lidar: torch.Tensor, labels_image: torch.Tensor,
                                       scores_lidar: torch.Tensor, scores_image: torch.Tensor,
                                       iou_thr: float = 0.5, mode: str = 'iou', conf_lambda: float = 0.1):
    if bboxes_lidar.shape[0] == 0:
        return torch.empty(size=(0,), dtype=torch.long)
    if bboxes_image.shape[0] == 0:
        return torch.zeros(size=(bboxes_lidar.shape[0],), dtype=torch.long)
    iou_calculator = BboxOverlaps2D()
    
    iou_matrix = iou_calculator(bboxes_lidar, bboxes_image, mode=mode)
    mask = labels_lidar.unsqueeze(1) != labels_image.unsqueeze(0)
    iou_matrix[mask] = 0
    
    if conf_lambda is not None:    
        confidence_weights = scores_lidar.unsqueeze(1).repeat(1, bboxes_image.shape[0])
        objective = -((1 - conf_lambda) * iou_matrix.cpu() - conf_lambda * confidence_weights)
    else:
        objective = -iou_matrix.cpu()
    
    lidar_ids, rgb_ids = linear_sum_assignment(objective)
    lidar_ids = bboxes_lidar.new_tensor(lidar_ids, dtype=torch.int)
    rgb_ids = bboxes_lidar.new_tensor(rgb_ids, dtype=torch.int)
    assigned_ious = iou_matrix[lidar_ids, rgb_ids]
    valid_matching = assigned_ious > iou_thr
    # print(lidar_ids, rgb_ids, assigned_ious, valid_matching)
    lidar_ids = lidar_ids[valid_matching]
    rgb_ids = rgb_ids[valid_matching]
    matching = bboxes_lidar.new_zeros(size=(bboxes_lidar.shape[0],), dtype=torch.int)
    matching[lidar_ids] = rgb_ids + 1
    return matching


def match_bboxes_iou_cls_groups(bboxes_lidar: torch.Tensor, bboxes_image: torch.Tensor, labels_lidar: torch.Tensor, 
                                labels_image: torch.Tensor, iou_thr: float = 0.5, mode: str = 'iou'):
    if bboxes_lidar.shape[0] == 0:
        return torch.empty(size=(0,), dtype=torch.long, device=bboxes_lidar.device)
    if bboxes_image.shape[0] == 0:
        return torch.zeros(size=(bboxes_lidar.shape[0],), dtype=torch.long, device=bboxes_lidar.device)
    iou_calculator = BboxOverlaps2D()
    
    mask = ((labels_lidar.unsqueeze(1) == 2) | (labels_image.unsqueeze(0) == 2)) & \
        (labels_lidar.unsqueeze(1) != labels_image.unsqueeze(0))
    
    iou_matrix = iou_calculator(bboxes_lidar, bboxes_image, mode=mode)
    iou_matrix[mask] = 0
    max_ious, max_indices = torch.max(iou_matrix, dim=1)
    matching = torch.where(max_ious >= iou_thr, max_indices + 1, 0)
    return matching


def assign_boxes(bboxes_gt: torch.Tensor, labels_gt: torch.Tensor, bboxes_pred: torch.Tensor, 
                 labels_pred: torch.Tensor, assigner: BaseAssigner, num_classes: int = 3):
    gt_data = InstanceData()
    gt_data.bboxes = bboxes_gt
    gt_data.labels = labels_gt
    
    pred_data = InstanceData()
    pred_data.priors = bboxes_pred
    
    return assigner.assign(pred_instances=pred_data, gt_instances=gt_data)


def filter_by_class(bboxes: torch.Tensor, labels: torch.Tensor, class_id):
    mask = labels == class_id
    return bboxes[mask], labels[mask]


def assign_boxes_by_class(bboxes_gt: torch.Tensor, labels_gt: torch.Tensor, bboxes_pred: torch.Tensor, 
                          labels_pred: torch.Tensor, assigner: BaseAssigner, num_classes: int = 3):
    all_assigned_gt_inds = []
    all_max_overlaps = []
    all_assigned_labels = []

    for class_id in range(num_classes):
        filtered_bboxes, _ = filter_by_class(bboxes_pred, labels_pred, class_id)
        filtered_gt_bboxes, filtered_gt_labels = filter_by_class(bboxes_gt, labels_gt, class_id)
        
        if filtered_gt_bboxes.numel() == 0 or filtered_bboxes.numel() == 0:
            assigned_gt_inds = torch.full((filtered_bboxes.size(0), ), -1, dtype=torch.long)
            max_overlaps = torch.zeros((filtered_bboxes.size(0), ))
            assigned_labels = torch.full((filtered_bboxes.size(0), ), -1, dtype=torch.long)
        else:
            assign_result = assign_boxes(filtered_gt_bboxes, filtered_gt_labels, filtered_bboxes, assigner)
            assigned_gt_inds = assign_result.gt_inds
            max_overlaps = assign_result.max_overlaps
            assigned_labels = assign_result.labels
        
        all_assigned_gt_inds.append(assigned_gt_inds)
        all_max_overlaps.append(max_overlaps)
        all_assigned_labels.append(assigned_labels)

    final_assigned_gt_inds = torch.full((bboxes_pred.size(0), ), -1, dtype=torch.long)
    final_max_overlaps = torch.zeros((bboxes_pred.size(0), ))
    final_assigned_labels = torch.full((bboxes_pred.size(0), ), -1, dtype=torch.long)

    for class_id in range(num_classes):
        class_mask = (labels_pred == class_id)
        final_assigned_gt_inds[class_mask] = all_assigned_gt_inds[class_id]
        final_max_overlaps[class_mask] = all_max_overlaps[class_id]
        final_assigned_labels[class_mask] = all_assigned_labels[class_id]
        
    return AssignResult(num_gts=filtered_gt_bboxes.shape[0], gt_inds=final_assigned_gt_inds, 
                        max_overlaps=final_max_overlaps, labels=final_assigned_labels)


def extract_corners_2d(bboxes_2d: torch.Tensor):   
    half_dimensions = (bboxes_2d[:, 2:4] - bboxes_2d[:, :2]).reshape(-1, 1, 2).repeat(1, 4, 1) / 2
    centers = bboxes_2d[:, 2:4].reshape(-1, 1, 2) - half_dimensions

    mult = [-1, 1]
    corners_multipliers = bboxes_2d.new_tensor([[x_mult, y_mult] for x_mult in mult for y_mult in mult])
    corners_relative = corners_multipliers * half_dimensions

    return corners_relative + centers


def get_frustums_from_2d_boxes(bboxes: torch.Tensor, P: torch.Tensor):
    K_inv = torch.linalg.inv(P[:, :3])
    corners_2d = extract_corners_2d(bboxes)
    K_inv = torch.unsqueeze(K_inv, 0).repeat(corners_2d.shape[0], 1, 1)
    corners_2d_hom = torch.cat([corners_2d, torch.ones_like(corners_2d[:, :, :1])], dim=-1)
    return torch.bmm(K_inv, corners_2d_hom.permute(0, 2, 1)).permute(0, 2, 1)


def clamp_boxes(bboxes, image_dim):
    bboxes[:, 0] = torch.clamp(bboxes[:, 0], min=0.0)
    bboxes[:, 1] = torch.clamp(bboxes[:, 1], min=0.0)
    bboxes[:, 2] = torch.clamp(bboxes[:, 2], min=0.0, max=image_dim[1])
    bboxes[:, 3] = torch.clamp(bboxes[:, 3], min=0.0, max=image_dim[0])
    return bboxes


def clamp_corners(corners, image_dim):
    corners[:, :, 0] = torch.clamp(corners[:, :, 0], min=0.0, max=image_dim[1])
    corners[:, :, 1] = torch.clamp(corners[:, :, 1], min=0.0, max=image_dim[0])
    return corners


def enlarge_bboxes_2d(bboxes, width_factor=0.2, height_factor=0.1):
    '''
    Enlarges bboxes in xyxy format by the specified factors
    '''
    widths = bboxes[..., 2] - bboxes[..., 0]
    heights = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] -= widths * width_factor / 2
    bboxes[..., 1] -= heights * height_factor / 2
    bboxes[..., 2] += widths * width_factor / 2
    bboxes[..., 3] += heights * height_factor / 2
    return bboxes


def get_camera_center(P: np.ndarray):
    camera_center = null_space(P).flatten()
    camera_center /= camera_center[-1]
    return camera_center


def get_frustum_bboxes(bboxes_left, bboxes_right, frustum_proposals, calibration_data,
                       labels_left, labels_rigth, scores_left, scores_right):
    cam_to_lidar = torch.linalg.inv(calibration_data['R0_rect'].cpu() @ calibration_data['Tr_velo_to_cam'].cpu())
    most_conf_image = torch.max(torch.cat([scores_left.unsqueeze(1), 
                                           scores_right.unsqueeze(1)], dim=1), dim=1)[1]
    labels = torch.gather(torch.cat([labels_left.unsqueeze(1), 
                                     labels_rigth.unsqueeze(1)], dim=1),
                          index=most_conf_image.unsqueeze(1), dim=1)

    frustum_left = get_frustums_from_2d_boxes(bboxes_left, calibration_data['P2'].cpu())
    frustum_right = get_frustums_from_2d_boxes(bboxes_right, calibration_data['P3'].cpu())

    camera_center_left = get_camera_center(calibration_data['P2'].cpu().numpy())
    camera_center_left = frustum_left.new_tensor(camera_center_left).unsqueeze(0).unsqueeze(0)
    infty_points_l = torch.cat([frustum_left * 100 + camera_center_left[:, :, :3], frustum_left.new_ones(*frustum_left.shape[:-1], 1)], dim=-1)
    cam_to_velo = torch.unsqueeze(cam_to_lidar, 0)
    camera_center_left = torch.bmm(cam_to_velo, camera_center_left.permute(0, 2, 1)).permute(0, 2, 1)
    camera_center_left[..., :3] /= camera_center_left[..., 3:]

    camera_center_right = get_camera_center(calibration_data['P3'].cpu().numpy())
    camera_center_right = frustum_right.new_tensor(camera_center_right).unsqueeze(0).unsqueeze(0)
    infty_points_r = torch.cat([frustum_right * 100 + camera_center_right[:, :, :3], frustum_right.new_ones(*frustum_right.shape[:-1], 1)], dim=-1)
    camera_center_right = torch.bmm(cam_to_velo, camera_center_right.permute(0, 2, 1)).permute(0, 2, 1)
    camera_center_right[..., :3] /= camera_center_right[..., 3:]

    cam_to_velo = torch.unsqueeze(cam_to_lidar, 0).repeat(infty_points_l.shape[0], 1, 1)
    infty_points_l = torch.bmm(cam_to_velo, infty_points_l.permute(0, 2, 1)).permute(0, 2, 1)
    infty_points_left = infty_points_l[:, (1, 3), :]
    # infty_points_left[..., :3] /= infty_points_left[..., 3:]
    cam_to_velo = torch.unsqueeze(cam_to_lidar, 0).repeat(infty_points_r.shape[0], 1, 1)
    infty_points_r = torch.bmm(cam_to_velo, infty_points_r.permute(0, 2, 1)).permute(0, 2, 1)
    infty_points_right = infty_points_r[:, (1, 3), :]
    # infty_points_right[..., :3] /= infty_points_right[..., 3:]

    a1 = infty_points_left[..., 1:2] - camera_center_left[..., 1:2]
    b1 = camera_center_left[..., 0:1] - infty_points_left[..., 0:1]
    c1 = a1 * camera_center_left[..., 0:1] + b1 * camera_center_left[..., 1:2]
    lines_bev_left = torch.cat([a1, b1, c1], dim=-1)

    a2 = infty_points_right[..., 1:2] - camera_center_right[..., 1:2]
    b2 = camera_center_right[..., 0:1] - infty_points_right[..., 0:1]
    c2 = a2 * camera_center_right[..., 0:1] + b2 * camera_center_right[..., 1:2]
    lines_bev_right = torch.cat([a2, b2, c2], dim=-1)

    intersections = torch.cross(lines_bev_left, lines_bev_right, dim=-1)
    intersections /= intersections[..., 2:]
    # lidar coords are right-handed, while camera coords are left-handed
    # but the cross-product intersection is defined for camera coords
    intersections = -intersections
    z = camera_center_left[..., 2] + infty_points_left[..., 2] * (intersections[..., 0] - camera_center_left[..., 0]) / infty_points_left[..., 0]
    intersections[..., 2] = z
    
    min_intersections_bev = intersections[..., :2].min(dim=1)[0]
    max_intersections_bev = intersections[..., :2].max(dim=1)[0]
    mean_intersection_vertical = intersections[..., 2].mean(dim=1)

    bboxes_3d_estimate = []
    indices = []
    for i, (z, min_bev, max_bev) in enumerate(zip(mean_intersection_vertical.numpy(), 
                                                  min_intersections_bev.numpy(),
                                                  max_intersections_bev.numpy())):
        points = frustum_proposals[i]
        bev_filter = (points[:, 0] >= min_bev[0]) & \
            (points[:, 0] <= max_bev[0]) & \
            (points[:, 1] >= min_bev[1]) & \
            (points[:, 1] <= max_bev[1])
        if bev_filter.sum() > 0:
            bev_center = points[bev_filter, :2].mean(dim=0)
            max_bev_points = points[bev_filter, :2].max(dim=0)[0].numpy()
            min_bev_points = points[bev_filter, :2].min(dim=0)[0].numpy()
            anchor = ANCHOR_SIZES[labels[i].item()]
            if max_bev_points[0] - min_bev_points[0] > max_bev_points[1] - min_bev_points[1]:
                yaw = 0 if anchor[0] > anchor[1] else np.pi / 2
            else:
                yaw = np.pi / 2 if anchor[0] > anchor[1] else 0
            z = points[bev_filter, 2].mean().item() - anchor[2] / 2
            indices.append(i)
            bbox = torch.cat([bev_center, torch.tensor([z]), torch.tensor(anchor), torch.tensor([yaw])])
            bboxes_3d_estimate.append(bbox)
        
            
    if len(indices) > 0:    
        bboxes_3d_estimate = torch.stack(bboxes_3d_estimate)   
        return (
            bboxes_3d_estimate, 
            -1 * bboxes_3d_estimate.new_ones((bboxes_3d_estimate.shape[0],), dtype=torch.float32), # dummy scores
            -1 * bboxes_3d_estimate.new_ones((bboxes_3d_estimate.shape[0],), dtype=torch.long), # dummy labels
            indices,
        )
    else:
        return (
            bboxes_left.new_empty((0, 7), dtype=torch.float32),
            bboxes_left.new_empty((0,), dtype=torch.float32),
            bboxes_left.new_empty((0,), dtype=torch.long),
            indices
        )
        
    
def get_road_intersection_depth(bboxes, road_plane, velo_to_cam, cam_to_img):
    # bottom_lines = get_frustums_from_2d_boxes(bboxes, cam_to_img.cpu())
    bottom_center = bboxes[:, (2,3)]
    bottom_center[:, 0] -= (bboxes[:, 2] - bboxes[:, 0]) / 2
    
    K_inv = torch.linalg.inv(cam_to_img[:, :3])
    bottom_center_hom = torch.cat([bottom_center, torch.ones_like(bottom_center[:, :1])], dim=-1)
    backprojections = bottom_center_hom @ K_inv.T
    
    camera_center = get_camera_center(cam_to_img.cpu().numpy())
    camera_center = backprojections.new_tensor(camera_center).unsqueeze(0)
    infty_points = torch.cat([backprojections * 100 + camera_center[:, :3], backprojections.new_ones(*backprojections.shape[:-1], 1)], dim=-1)
    
    directions = infty_points[:, :3] - camera_center[:, :3]
    directions /= directions.norm(2, dim=-1, keepdim=True)
    
    t = - ((road_plane[:3] * camera_center[:, :3]).sum(dim=-1) + road_plane[3]) / (road_plane[:3] * directions).sum(dim=-1)
    intersections = camera_center[:, :3] + directions * t.unsqueeze(1).repeat(1, 3)
    intersections = torch.cat([intersections, intersections.new_ones(*intersections.shape[:-1], 1)], dim=-1) 

    intersections = intersections @ torch.linalg.inv(velo_to_cam).T
    intersections[:, :3] /= intersections[:, 3:]
    return intersections


def assign_with_epipolar_lines(bboxes_left: Tensor, bboxes_right: Tensor, 
                               labels_left: Tensor, labels_right: Tensor, 
                               fundamental_matrix: Tensor, **kwargs):
    if bboxes_left.shape[0] == 0 or bboxes_right.shape[0] == 0:
        return bboxes_left.new_empty((0,), dtype=int), bboxes_left.new_empty((0,), dtype=int)
    corners_right_2d = extract_corners_2d(bboxes_right)
    corners_right_2d = torch.cat([corners_right_2d, torch.ones_like(corners_right_2d[:, :, :1])], dim=-1)

    corners_left_2d = extract_corners_2d(bboxes_left)
    corners_left_2d = torch.cat([corners_left_2d, torch.ones_like(corners_left_2d[:, :, :1])], dim=-1)
    
    F = fundamental_matrix.unsqueeze(0).expand(corners_left_2d.shape[0], -1, -1)
    epipolar_lines = torch.matmul(corners_left_2d, F.transpose(1, 2))
    epipolar_lines /= epipolar_lines[:, :, 2:]
    
    # computes the 'distances' between each pair of bounding boxes
    # a distance is the sum of the distances between each pair of (epipolar line, corner)
    dot_products = torch.abs(torch.diagonal(torch.einsum('nik,mjk->nmij', epipolar_lines, corners_right_2d), dim1=2, dim2=3))
    denom = torch.sqrt(torch.sum(epipolar_lines[:, :, :2]**2, dim=2))
    distances = torch.sum(dot_products / denom.unsqueeze(1), dim=2) # from shape (N, N, 4) to (N, N)
    
    mask = ((labels_left.unsqueeze(1) == 2) | (labels_right.unsqueeze(0) == 2)) & \
        (labels_left.unsqueeze(1) != labels_right.unsqueeze(0))
    distances[mask] = distances.max() + 1

    row_ids, col_ids = linear_sum_assignment(distances.cpu())
    return labels_left.new_tensor(row_ids), labels_right.new_tensor(col_ids)
    

def project_and_filter_boxes(bboxes_3d, scores_3d, labels_3d, calibration_data, img_dim_left, img_dim_right, score_threshold=0.5):
    bboxes_proj_left = project_bboxes(bboxes_3d, calibration_data['P2'].cpu(), lidar=True, 
                                      Tr_velo_to_cam=calibration_data['Tr_velo_to_cam'].cpu(), 
                                      R_rect=calibration_data['R0_rect'].cpu())
    bboxes_proj_left = clamp_boxes(bboxes_proj_left, img_dim_left)
    inside_image_left = (bboxes_proj_left[:, 0] < bboxes_proj_left[:, 2]) & (bboxes_proj_left[:, 1] < bboxes_proj_left[:, 3])
    bboxes_proj_right = project_bboxes(bboxes_3d, calibration_data['P3'].cpu(), lidar=True, 
                                       Tr_velo_to_cam=calibration_data['Tr_velo_to_cam'].cpu(), 
                                       R_rect=calibration_data['R0_rect'].cpu())
    bboxes_proj_right = clamp_boxes(bboxes_proj_right, img_dim_right)
    inside_image_right = (bboxes_proj_right[:, 0] < bboxes_proj_right[:, 2]) & (bboxes_proj_right[:, 1] < bboxes_proj_right[:, 3])
    # filtering 3d boxes that do not project in the image
    keep_boxes = (scores_3d > score_threshold) & (inside_image_left | inside_image_right)
    return (
        bboxes_proj_left[keep_boxes], 
        bboxes_proj_right[keep_boxes], 
        bboxes_3d[keep_boxes], 
        scores_3d[keep_boxes], 
        labels_3d[keep_boxes], 
        keep_boxes
    )