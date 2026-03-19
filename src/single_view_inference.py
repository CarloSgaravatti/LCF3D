import numpy as np
from pathlib import Path
from typing_extensions import Tuple, List, Dict, Union, Sequence
from copy import deepcopy
from PIL import Image
import torch
from torch import Tensor
import numpy as np
import warnings
from bbox_utils import *
from visualization.visualization_utils import *
from base_inference import BaseLateFusionInferencer

from mmdet.models.task_modules import BboxOverlaps2D
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import box3d_multiclass_nms, nms_bev
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from scipy.optimize import linear_sum_assignment
from mmcv.ops import box_iou_rotated
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import networkx as nx


class LateFusionSingleViewInferencer(BaseLateFusionInferencer):
    """
    Late Fusion inference class for single view setup with switchable matching strategies.

    Args:
        late_fusion_cfg (Union[Dict, str, Path]): Configuration dict for late fusion.
            If a file path is passed, it will load the configuration from it
        cam_to_img (Union[np.ndarray, Tensor, Sequence[Sequence[float]]], optional): 
            Transformation matrix from camera to image coordinates.
        lidar_to_cam (Union[np.ndarray, Tensor, Sequence[Sequence[float]]], optional): 
            Transformation matrix from LiDAR to camera coordinates.
        device (Union[str, torch.device], optional): Device to run the models on. Defaults to 'cuda:0'.
    """
    def __init__(self,
                 late_fusion_cfg: Union[Dict, str, Path],
                 cam_to_img: Union[np.ndarray, Tensor, Sequence[Sequence[float]]] = None,
                 lidar_to_cam: Union[np.ndarray, Tensor, Sequence[Sequence[float]]] = None,
                 device: Union[str, torch.device] = 'cuda:0'):
        
        # Call parent constructor
        super().__init__(late_fusion_cfg, device)

        self.use_clustering = self.late_fusion_cfg.get('use_clustering', False)
                
        self.lidar_to_cam = lidar_to_cam
        if lidar_to_cam is None:
            warnings.warn('lidar_to_cam is None, if not passed sample by sample the results will be not consistent')
            self.lidar_to_cam = torch.eye(4, dtype=torch.float32)    
        elif not isinstance(lidar_to_cam, Tensor):
            self.lidar_to_cam = torch.tensor(lidar_to_cam, dtype=torch.float32)
        self.lidar_to_cam = self.lidar_to_cam.to(self.device)
        
        self.cam_to_img = cam_to_img
        if cam_to_img is None:
            warnings.warn('cam_to_img is None, if not passed sample by sample the results will be not consistent')
            self.cam_to_img = torch.eye(4, dtype=torch.float32)[:3, :]  
        elif not isinstance(cam_to_img, Tensor):
            self.cam_to_img = torch.tensor(cam_to_img, dtype=torch.float32)[:3, :]
        self.cam_to_img = self.cam_to_img.to(self.device)
        
        # Additional single-view specific configs
        self.num_classes = self.late_fusion_cfg.get('num_classes', 3)
        self.classes = self.late_fusion_cfg.get('classes', [''] * self.num_classes)
        self.class_dict = {i: self.classes[i] for i in range(self.num_classes)}
        
        self.bb_match_iou_thr = self.late_fusion_cfg.get('bbox_matching_iou_thr', 0.5)
        self.recovery_iou_thr = self.late_fusion_cfg.get('detection_recovery_iou_thr', 0.4)
        self.bb_match_mode = self.late_fusion_cfg.get('bbox_matching_mode', 'iou')
        self.min_pts_frustum = self.late_fusion_cfg.get('min_pts_frustum', 10)
        
        self.use_score_fusion = self.late_fusion_cfg.get('use_score_fusion', True)
        self.class_priors = self.late_fusion_cfg.get('class_prior', [1 / self.num_classes for _ in range(self.num_classes)])
        self.class_priors = torch.tensor(self.class_priors, dtype=torch.float32, device=self.device)
        self.use_final_nms = self.late_fusion_cfg.get('use_final_nms', True)
        self.final_nms_cfg = self.late_fusion_cfg.get('final_nms_cfg', {})
        
        # Clustering-specific parameters
        self.dbscan_eps = self.late_fusion_cfg.get('dbscan_eps', 0.5)
        self.confidence_lambda = self.late_fusion_cfg.get('confidence_lambda', None)
        self.match_class = self.late_fusion_cfg.get('match_class', False)
        # Clustering method and thresholds.
        # Supported methods:
        # - connected_components: partitions graph components
        # - cliques: uses maximal cliques (can overlap)
        self.clustering_method = self.late_fusion_cfg.get('clustering_method', 'connected_components')
        # Backward-compat: if config uses boolean use_cliques, honor it.
        if 'use_cliques' in self.late_fusion_cfg:
            self.clustering_method = 'cliques' if self.late_fusion_cfg.get('use_cliques', False) else 'connected_components'

        self.cluster_bev_iou_thr_cc = self.late_fusion_cfg.get(
            'cluster_bev_iou_thr_cc',
            self.late_fusion_cfg.get('cluster_bev_iou_thr', 0.5),
        )
        self.cluster_bev_iou_thr_clique = self.late_fusion_cfg.get(
            'cluster_bev_iou_thr_clique',
            self.late_fusion_cfg.get('cluster_bev_iou_thr', 0.3),
        )
        
    def bbox_matching(self, bboxes_3d, scores_3d, labels_3d, corners_3d, bboxes_2d, 
                      scores_2d, labels_2d, img_shape, lidar_to_cam, cam_to_img):
        """
        Match 3D bboxes with 2D detections using either frustum or clustering strategy.
        """
        # Project 3D bboxes to image
        corners_cam, corners_proj = corners_to_img_coord(
            corners_3d, P=cam_to_img, lidar=True, T=lidar_to_cam)
        
        # Seeing if the bounding boxes are in the field of view of the image
        corners_inside = (corners_proj[:, :, 0] >= 0) & \
            (corners_proj[:, :, 0] < img_shape[1]) & \
            (corners_proj[:, :, 1] >= 0) & \
            (corners_proj[:, :, 1] < img_shape[0])
        corners_inside = corners_inside.sum(dim=1) > 0
        front_view_filter = torch.max(corners_cam[:, :, 2], dim=1)[0] > 0
        front_view_filter = front_view_filter & corners_inside
        
        corners_proj = clamp_corners(corners_proj, img_shape)
        bboxes_proj = axis_aligned_bboxes(corners_proj)
        
        inside_image = front_view_filter & \
            (bboxes_proj[:, 0] < bboxes_proj[:, 2]) & \
            (bboxes_proj[:, 1] < bboxes_proj[:, 3])
        
        bboxes_3d_valid = bboxes_3d[inside_image]
        scores_3d_valid = scores_3d[inside_image]
        labels_3d_valid = labels_3d[inside_image]
        bboxes_proj_valid = bboxes_proj[inside_image]
        
        if self.use_clustering:
            return self._bbox_matching_clustering(
                bboxes_3d_valid, scores_3d_valid, labels_3d_valid, 
                bboxes_proj_valid, bboxes_2d, scores_2d, labels_2d, 
                bboxes_3d, scores_3d, labels_3d, inside_image
            )
        else:
            return self._bbox_matching_linear_assignment(
                bboxes_3d_valid, scores_3d_valid, labels_3d_valid, 
                bboxes_proj_valid, bboxes_2d, scores_2d, labels_2d,
                bboxes_3d, scores_3d, labels_3d, inside_image
            )
    
    def _bbox_matching_linear_assignment(self, bboxes_3d_valid, scores_3d_valid, labels_3d_valid, 
                                         bboxes_proj_valid, bboxes_2d, scores_2d, labels_2d,
                                         bboxes_3d, scores_3d, labels_3d, inside_image):
        """Matching using linear sum assignment."""
        
        assignment = match_bboxes_linear_sum_assign(
            self.iou_calculator, bboxes_proj_valid, bboxes_2d, 
            scores_lidar=scores_3d_valid, scores_image=scores_2d,
            mode=self.bb_match_mode, iou_thr=self.bb_match_iou_thr)
        
        # Get unmatched 2D boxes
        unmatched_2d = torch.isin(
            torch.arange(bboxes_2d.shape[0]).to(bboxes_proj_valid.device), 
            assignment - 1, invert=True)
        
        matches_mask = torch.where(assignment >= 1, True, False)
        
        return {
            'bboxes_3d': bboxes_3d_valid[matches_mask],
            'scores_3d': scores_3d_valid[matches_mask],
            'labels_3d': labels_3d_valid[matches_mask],
            'bboxes_2d': bboxes_2d[assignment[matches_mask] - 1] if matches_mask.sum() > 0 else bboxes_2d.new_empty((0, 4)),
            'scores_2d': scores_2d[assignment[matches_mask] - 1] if matches_mask.sum() > 0 else scores_2d.new_empty((0,)),
            'labels_2d': labels_2d[assignment[matches_mask] - 1] if matches_mask.sum() > 0 else labels_2d.new_empty((0,)),
            'unmatched_2d': bboxes_2d[unmatched_2d],
            'unmatched_2d_scores': scores_2d[unmatched_2d],
            'unmatched_2d_labels': labels_2d[unmatched_2d],
            'oov_bboxes_3d': bboxes_3d[~inside_image],
            'oov_scores_3d': scores_3d[~inside_image],
            'oov_labels_3d': labels_3d[~inside_image],
        }
    
    def _bbox_matching_clustering(self, bboxes_3d_valid, scores_3d_valid, labels_3d_valid, 
                                 bboxes_proj_valid, bboxes_2d, scores_2d, labels_2d, 
                                 bboxes_3d, scores_3d, labels_3d, inside_image):
        """Clustering-based matching.

        The clustering graph is built in BEV using rotated IoU between 3D boxes.
        Clusters can be formed using connected components or maximal cliques.
        """
        if bboxes_3d_valid.shape[0] == 0:
            return {
                'bboxes_3d': bboxes_3d_valid.new_empty((0, 7)),
                'scores_3d': scores_3d_valid.new_empty((0,)),
                'labels_3d': labels_3d_valid.new_empty((0,), dtype=torch.long),
                'bboxes_2d': bboxes_2d.new_empty((0, 4)),
                'scores_2d': scores_2d.new_empty((0,)),
                'labels_2d': labels_2d.new_empty((0,), dtype=torch.long),
                'unmatched_2d': bboxes_2d,
                'unmatched_2d_scores': scores_2d,
                'unmatched_2d_labels': labels_2d,
                'oov_bboxes_3d': bboxes_3d[~inside_image],
                'oov_scores_3d': scores_3d[~inside_image],
                'oov_labels_3d': labels_3d[~inside_image],
            }
        
        # Convert to box type for BEV operations
        box_3d_obj = self.box_type_3d(bboxes_3d_valid)
        iou_matrix_bev = box_iou_rotated(box_3d_obj.bev, box_3d_obj.bev, aligned=False, mode='iou')
        diff_classes = labels_3d_valid[:, None] != labels_3d_valid[None, :]
        iou_matrix_bev[diff_classes] = 0
        
        if self.clustering_method == 'cliques':
            graph = iou_matrix_bev > self.cluster_bev_iou_thr_clique
            sparse_matrix = sp.csr_matrix(graph.cpu())
            G = nx.from_scipy_sparse_array(sparse_matrix)

            cliques = list(nx.find_cliques(G))
            # NOTE: cliques can overlap -> we treat each clique as a "cluster".
            cluster_labels = sum([[i] * len(clique) for i, clique in enumerate(cliques)], [])
            bboxes_ids = sum(cliques, [])
            clusters_torch = torch.tensor(cluster_labels, dtype=torch.long, device=iou_matrix_bev.device)
            num_clusters = len(cliques)
        else:
            # Connected components clustering using BEV IoU threshold
            graph = iou_matrix_bev > self.cluster_bev_iou_thr_cc
            num_clusters, clusters = connected_components(graph.cpu().numpy(), directed=False)
            clusters_torch = torch.tensor(clusters, dtype=torch.long, device=iou_matrix_bev.device)
            bboxes_ids = list(range(clusters_torch.shape[0]))
        
        # Match clusters to 2D boxes
        bboxes_proj_for_match = bboxes_proj_valid[bboxes_ids]
        iou_matrix_2d = self.iou_calculator(bboxes_proj_for_match, bboxes_2d, mode='iou')
        reduced_iou = iou_matrix_2d.new_zeros((num_clusters, bboxes_2d.shape[0]))
        reduced_iou = reduced_iou.scatter_reduce_(
            0, clusters_torch.unsqueeze(1).expand(-1, bboxes_2d.shape[0]), 
            iou_matrix_2d, reduce='amax', include_self=False)
        
        lidar_ids, rgb_ids = linear_sum_assignment(-reduced_iou.cpu().numpy())
        lidar_ids = torch.tensor(lidar_ids, dtype=torch.long, device=iou_matrix_2d.device)
        rgb_ids = torch.tensor(rgb_ids, dtype=torch.long, device=iou_matrix_2d.device)
        assigned_ious = reduced_iou[lidar_ids, rgb_ids]
        valid_matching = assigned_ious > self.bb_match_iou_thr
        lidar_ids = lidar_ids[valid_matching]
        rgb_ids = rgb_ids[valid_matching]
        
        cluster_matching = torch.zeros(num_clusters, dtype=torch.long, device=iou_matrix_bev.device)
        cluster_matching[lidar_ids] = rgb_ids + 1
        
        # Get cluster representatives (highest confidence)
        scores_for_match = scores_3d_valid[bboxes_ids]
        cluster_sortidx = torch.argsort(clusters_torch)
        cluster_ids, cluster_counts = torch.unique_consecutive(clusters_torch[cluster_sortidx], return_counts=True)
        
        end_indices = torch.cumsum(cluster_counts, dim=0).cpu().tolist()
        start_indices = [0] + end_indices[:-1]
        
        max_indices = torch.zeros(num_clusters, dtype=torch.long, device=iou_matrix_bev.device)
        for cluster_id, a, b in zip(cluster_ids, start_indices, end_indices):
            indices = cluster_sortidx[a:b]
            max_indices[cluster_id] = indices[torch.argmax(scores_for_match[indices], dim=0)]

        # Map representative indices back to original 3D box indices
        max_indices_orig = max_indices.new_tensor([bboxes_ids[i] for i in max_indices.tolist()])
        
        bboxes_3d_max = bboxes_3d_valid[max_indices_orig]
        labels_3d_max = labels_3d_valid[max_indices_orig]
        scores_3d_max = scores_3d_valid[max_indices_orig]
        
        matches_mask = cluster_matching > 0
        matched_2d_ids = (cluster_matching[matches_mask] - 1).long()
        
        # Unmatched 2D boxes
        unmatched_2d = torch.ones(bboxes_2d.shape[0], dtype=torch.bool)
        unmatched_2d[matched_2d_ids] = False
        
        return {
            'bboxes_3d': bboxes_3d_max[matches_mask],
            'scores_3d': scores_3d_max[matches_mask],
            'labels_3d': labels_3d_max[matches_mask],
            'bboxes_2d': bboxes_2d[matched_2d_ids] if matches_mask.sum() > 0 else bboxes_2d.new_empty((0, 4)),
            'scores_2d': scores_2d[matched_2d_ids] if matches_mask.sum() > 0 else scores_2d.new_empty((0,)),
            'labels_2d': labels_2d[matched_2d_ids] if matches_mask.sum() > 0 else labels_2d.new_empty((0,)),
            'unmatched_2d': bboxes_2d[unmatched_2d],
            'unmatched_2d_scores': scores_2d[unmatched_2d],
            'unmatched_2d_labels': labels_2d[unmatched_2d],
            'oov_bboxes_3d': bboxes_3d[~inside_image],
            'oov_scores_3d': scores_3d[~inside_image],
            'oov_labels_3d': labels_3d[~inside_image],
        }
    
    def semantic_fusion(self, bboxes_3d, scores_3d, labels_3d, bboxes_2d, scores_2d, labels_2d):
        """Fuse semantic information from 2D detections with 3D detections."""
        if bboxes_3d.shape[0] == 0 or bboxes_2d.shape[0] == 0:
            return labels_3d, scores_3d
        
        different_labels = labels_3d != labels_2d
        
        new_labels_3d = labels_3d.clone()
        if self.use_label_fusion:
            new_labels_3d[different_labels] = labels_2d[different_labels]
        
        new_scores_3d = scores_3d.clone()
        if self.use_score_fusion:
            new_scores_3d[different_labels] = scores_2d[different_labels]
            new_scores_3d[~different_labels] = (new_scores_3d[~different_labels] * scores_2d[~different_labels]) / self.class_priors[labels_2d[~different_labels]]
        return new_labels_3d, new_scores_3d
    
    def detection_recovery(self, collate_data: Dict, unmatched_bboxes_2d: Tensor, unmatched_scores_2d: Tensor, 
                          unmatched_labels_2d: Tensor, lidar_to_cam: Tensor, cam_to_img: Tensor):
        """
        Recover missed detections using frustum-based proposals (frustum RPN).
        Based on expert_late_fusion_final_single_view_frustum_v3.py::frustum_2d_rpn
        """
        if unmatched_bboxes_2d.shape[0] == 0 or self.frustum_detector is None:
            return {
                'bboxes_3d': torch.empty((0, 7), dtype=torch.float32, device=self.device),
                'scores_3d': torch.empty((0,), dtype=torch.float32, device=self.device),
                'labels_3d': torch.empty((0,), dtype=torch.long, device=self.device),
            }
        
        bboxes_left_enlarge = enlarge_bboxes_2d(unmatched_bboxes_2d.clone(), self.enlarge_factor, self.enlarge_factor)
        ori_left = unmatched_bboxes_2d.cpu().numpy()
        one_hot_vectors = torch.nn.functional.one_hot(unmatched_labels_2d.long(), num_classes=self.valid_2d_classes.shape[0])
        
        scan = collate_data['inputs']['points'][0].to(lidar_to_cam.device)
        scan = scan[scan[:, 0] > 0]

        scan_for_frustum = scan
        if getattr(self, 'use_dims_frustum', None) is not None:
            if len(self.use_dims_frustum) < 3 or self.use_dims_frustum[:3] != [0, 1, 2]:
                raise ValueError('use_dims_frustum must start with [0, 1, 2] so frustum_pc[:, :3] is xyz.')
            scan_for_frustum = scan[:, self.use_dims_frustum]

        points = scan[:, 0:3]
        velo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        
        coord_left = cam_to_img.matmul(lidar_to_cam.matmul(velo.t()))
        coord_left[:2] /= coord_left[2, :]
        coord_left[2] = 1
        coord_left = coord_left.T
        
        if self.align_frustum:
            K_inv_left = np.linalg.inv(cam_to_img.cpu().numpy()[:, :3])
            cam_to_lidar = np.linalg.inv(lidar_to_cam.cpu().numpy())
        
        new_bboxes_3d = []
        new_scores_3d = []
        new_labels_3d = []
        indices_2d = []
        rt_matrices = []
        yaw_angles = []
        frustum_proposals = {'inputs': {'points': []}, 'data_samples': []}
        
        # Extract frustum proposals for each 2D box
        for i, left_box in enumerate(bboxes_left_enlarge.cpu().numpy()):
            fov_inds = (coord_left[:, 0] <= left_box[2]) & \
                       (coord_left[:, 0] >= left_box[0]) & \
                       (coord_left[:, 1] <= left_box[3]) & \
                       (coord_left[:, 1] >= left_box[1])
            
            if torch.sum(fov_inds) > 10:
                indices_2d.append(i)
                data_sample = deepcopy(collate_data['data_samples'][0])
                metainfo = data_sample.metainfo
                metainfo['one_hot_vector'] = one_hot_vectors[i, :]
                data_sample.set_metainfo(metainfo)
                
                frustum_pc = scan_for_frustum[fov_inds, :].clone()
                
                # Add Gaussian likelihoods if enabled
                if self.use_gaussian_likelihoods:
                    wl = ori_left[i, 2] - ori_left[i, 0]
                    hl = ori_left[i, 3] - ori_left[i, 1]
                    xl = ori_left[i, 0] + wl / 2
                    yl = ori_left[i, 1] + hl / 2
                    likelihoods = torch.exp(
                        -((coord_left[fov_inds, 0] - xl) ** 2 / (2 * wl ** 2)) -
                        ((coord_left[fov_inds, 1] - yl) ** 2 / (2 * hl ** 2)))
                    frustum_pc = torch.cat([frustum_pc, likelihoods.unsqueeze(1)], dim=-1)
                
                # Align frustum if enabled
                if self.align_frustum:
                    left_center = np.concatenate(
                        [(left_box[2:4] + left_box[0:2]) / 2, [1]])[:, np.newaxis]
                    backprojection = np.append(
                        K_inv_left.dot(left_center).flatten(), [0])[np.newaxis, :]
                    backprojection = cam_to_lidar.dot(backprojection.T).T
                    yaw_lidar = np.arctan2(backprojection[:, 1], backprojection[:, 0])[0]
                    
                    rotation_matrix = scan.new_tensor([
                        [np.cos(-yaw_lidar), -np.sin(-yaw_lidar), 0],
                        [np.sin(-yaw_lidar), np.cos(-yaw_lidar), 0],
                        [0, 0, 1],
                    ])
                    frustum_pc[:, :3] = rotation_matrix.matmul(frustum_pc[:, :3].t()).t()
                    rt_matrices.append(rotation_matrix)
                    yaw_angles.append(yaw_lidar)
                
                frustum_proposals['inputs']['points'].append(frustum_pc)
                frustum_proposals['data_samples'].append(data_sample)
        
        # Process frustum proposals through frustum detector
        if len(indices_2d) > 0:
            bboxes_3d = []
            scores_3d = []
            labels_3d = []
            
            for i in range(len(indices_2d)):
                proposal = {
                    'data_samples': [frustum_proposals['data_samples'][i]],
                    'inputs': {'points': [frustum_proposals['inputs']['points'][i]]}
                }
                
                with torch.no_grad():
                    detection_output = self.frustum_detector.test_step(proposal)
                
                # Handle different output formats
                if len(detection_output) == 4:
                    _, new_bboxes_3d, new_scores_3d, new_labels_3d = detection_output
                    scores_3d.append(new_scores_3d)
                    labels_3d.append(new_labels_3d)
                elif len(detection_output) == 3:
                    _, new_bboxes_3d, new_scores_3d = detection_output
                    scores_3d.append(new_scores_3d)
                else:
                    _, new_bboxes_3d = detection_output
                
                # De-align frustum if needed
                if self.align_frustum:
                    new_bboxes_3d[:, :3] = torch.linalg.inv(
                        rt_matrices[i]).matmul(new_bboxes_3d[:, :3].t()).t()
                    new_bboxes_3d[:, -1] += yaw_angles[i]
                
                bboxes_3d.append(new_bboxes_3d.cpu())
            
            # Aggregate predictions from frustum proposals
            new_bboxes_3d = torch.cat(bboxes_3d, dim=0)[:, :7]
            new_bboxes_3d[:, 2] += new_bboxes_3d[:, 5] / 2  # Center at centroid
            new_bboxes_3d = new_bboxes_3d.to(cam_to_img.device)
            
            if len(scores_3d) > 0:
                new_scores_3d = torch.cat(scores_3d, dim=0).squeeze(1)
            elif self.use_label_fusion:
                new_scores_3d = -new_bboxes_3d.new_ones((new_bboxes_3d.shape[0],), dtype=torch.float32)
            else:
                new_scores_3d = unmatched_scores_2d[indices_2d]
            
            if len(labels_3d) > 0:
                new_labels_3d = torch.cat(labels_3d, dim=0).squeeze(1)
            elif self.use_label_fusion:
                new_labels_3d = -new_bboxes_3d.new_ones((new_bboxes_3d.shape[0],), dtype=torch.long)
            else:
                new_labels_3d = unmatched_labels_2d[indices_2d]
            
            new_left_boxes = unmatched_bboxes_2d[indices_2d].to(cam_to_img.device)
            
            # Project 3D bboxes back to image and filter by IoU
            bboxes_proj_left = project_bboxes(
                new_bboxes_3d, P=cam_to_img, lidar=True, T=lidar_to_cam)
            
            ious_left = BboxOverlaps2D()(bboxes_proj_left, new_left_boxes, is_aligned=True)
            iou_filter = ious_left > self.recovery_iou_thr
            
            new_bboxes_3d = new_bboxes_3d[iou_filter]
            new_scores_3d = new_scores_3d[iou_filter]
            new_labels_3d = new_labels_3d[iou_filter]
            new_left_boxes = new_left_boxes[iou_filter].to(cam_to_img.device)
            
            num_valid = torch.sum(iou_filter)
            
            # Apply semantic fusion on recovered detections
            if self.use_label_fusion and num_valid > 0:
                new_labels_3d, new_scores_3d = self.semantic_fusion(
                    new_bboxes_3d, new_scores_3d, new_labels_3d,
                    new_left_boxes, unmatched_scores_2d[indices_2d][iou_filter].to(cam_to_img.device),
                    unmatched_labels_2d[indices_2d][iou_filter].to(cam_to_img.device)
                )
                new_scores_3d = new_scores_3d * ious_left[iou_filter]
            elif num_valid > 0:
                new_scores_3d = new_scores_3d * (ious_left[iou_filter] ** 2)
        else:
            # No proposals generated
            new_bboxes_3d = unmatched_bboxes_2d.new_empty((0, 7), dtype=torch.float32)
            new_scores_3d = unmatched_scores_2d.new_empty((0,), dtype=torch.float32)
            new_labels_3d = unmatched_labels_2d.new_empty((0,), dtype=torch.long)
        
        # Shift back to mmdetection3d format (center at bottom)
        new_bboxes_3d[:, 2] -= new_bboxes_3d[:, 5] / 2
        
        return {
            'bboxes_3d': new_bboxes_3d,
            'scores_3d': new_scores_3d,
            'labels_3d': new_labels_3d,
        }
        
    def predict(self, img_file: str, pc_file: str,
                lidar_to_cam: Tensor = None, cam_to_img: Tensor = None,
                points: np.ndarray = None) -> InstanceData:
        """
        Runs inference on the provided image and LiDAR files and returns the final detections.

        Args:
            img_file (str): Path to the image file.
            pc_file (str): Path to the LiDAR file (point cloud).
            lidar_to_cam (Tensor, optional): Transformation matrix from LiDAR to camera coordinates.
            cam_to_img (Tensor, optional): Transformation matrix from camera to image coordinates.

        Returns:
            InstanceData: The final detections including 3d bounding boxes, scores, and labels.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img is None:
            cam_to_img = self.cam_to_img
        
        # Run branch inferences
        rgb_results = self.rgb_branch_inference(img_file)
        bboxes_2d, labels_2d, scores_2d, img_shape, _ = rgb_results[0]  # _ = masks (ignored)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud, collate_data = self.lidar_branch_inference(
            pc_file, points=points
        )
        
        # Perform bounding box matching with chosen strategy
        bbox_matching_dict = self.bbox_matching(bboxes_3d, scores_3d, labels_3d, corners_3d,
                                                bboxes_2d, scores_2d, labels_2d,
                                                img_shape, lidar_to_cam, cam_to_img)
        
        matching = {
            'bboxes_3d': bbox_matching_dict['bboxes_3d'],
            'scores_3d': bbox_matching_dict['scores_3d'],
            'labels_3d': bbox_matching_dict['labels_3d'],
            'bboxes_2d': bbox_matching_dict['bboxes_2d'],
            'scores_2d': bbox_matching_dict['scores_2d'],
            'labels_2d': bbox_matching_dict['labels_2d'],
        }
        
        # Apply semantic fusion if enabled
        if self.use_label_fusion and matching['bboxes_3d'].shape[0] > 0:
            new_labels_3d, new_scores_3d = self.semantic_fusion(
                matching['bboxes_3d'], matching['scores_3d'], matching['labels_3d'],
                matching['bboxes_2d'], matching['scores_2d'], matching['labels_2d']
            )
            matching['labels_3d'] = new_labels_3d
            matching['scores_3d'] = new_scores_3d
        
        # Detection recovery for unmatched 2D boxes
        if self.use_detection_recovery and self.frustum_detector is not None:
            recovery_output = self.detection_recovery(
                collate_data, bbox_matching_dict['unmatched_2d'], 
                bbox_matching_dict['unmatched_2d_scores'], bbox_matching_dict['unmatched_2d_labels'],
                lidar_to_cam, cam_to_img
            )
            matching['bboxes_3d'] = torch.cat([matching['bboxes_3d'], recovery_output['bboxes_3d']], dim=0)
            matching['scores_3d'] = torch.cat([matching['scores_3d'], recovery_output['scores_3d']], dim=0)
            matching['labels_3d'] = torch.cat([matching['labels_3d'], recovery_output['labels_3d']], dim=0)
        
        bboxes_3d_final = matching['bboxes_3d']
        scores_3d_final = matching['scores_3d']
        labels_3d_final = matching['labels_3d']
        
        # Keep out-of-view boxes if enabled
        if self.late_fusion_cfg.get('keep_oov_bboxes', False):
            bboxes_3d_final = torch.cat([bboxes_3d_final, bbox_matching_dict['oov_bboxes_3d']], dim=0)
            scores_3d_final = torch.cat([scores_3d_final, bbox_matching_dict['oov_scores_3d']], dim=0)
            labels_3d_final = torch.cat([labels_3d_final, bbox_matching_dict['oov_labels_3d']], dim=0)

        # Apply final NMS if enabled
        if self.use_final_nms and bboxes_3d_final.shape[0] > 0:
            nms_cfg = ConfigDict(use_rotate_nms=True, nms_thr=self.final_nms_cfg.get('thresh', 0.01))
            score_thr = self.final_nms_cfg.get('score_thr', 0.0001)
            bev_boxes_for_nms = xywhr2xyxyr(self.box_type_3d(bboxes_3d_final).bev)
            scores_for_nms = bboxes_3d_final.new_zeros((scores_3d_final.shape[0], self.num_classes), dtype=torch.float32)
            scores_for_nms[torch.arange(scores_3d_final.shape[0], dtype=torch.long), labels_3d_final.long()] = scores_3d_final
            bboxes_3d_final, scores_3d_final, labels_3d_final = box3d_multiclass_nms(
                bboxes_3d_final, bev_boxes_for_nms, scores_for_nms,
                score_thr=score_thr, max_num=10000, cfg=nms_cfg)
        
        # Create final detections instance
        final_detections = InstanceData()
        final_detections.bboxes_3d = self.box_type_3d(bboxes_3d_final)
        final_detections.scores_3d = scores_3d_final
        final_detections.labels_3d = labels_3d_final
        
        return final_detections
    
    def visualize_predict(self, img_file: str, pc_file: str, save_path: Union[str, Path],
                          lidar_to_cam: Tensor = None, cam_to_img: Tensor = None) -> InstanceData:
        """
        Runs inference and saves visualizations of detections to the specified path.

        Args:
            img_file (str): Path to the image file.
            pc_file (str): Path to the LiDAR file (point cloud).
            save_path (Union[str, Path]): Path to save the following visualization images
                - rgb_detections.png: image containing the 2d detections from the rgb branch
                - lidar_detections.png: image containing the 3d detections from the lidar branch
                - fusion_detections.png: image containing the final 3d detections
            lidar_to_cam (Tensor, optional): Transformation matrix from LiDAR to camera coordinates.
            cam_to_img (Tensor, optional): Transformation matrix from camera to image coordinates.

        Returns:
            InstanceData: The final detections including bounding boxes, scores, and labels.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img is None:
            cam_to_img = self.cam_to_img
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Run branch inferences
        rgb_results = self.rgb_branch_inference(img_file)
        bboxes_2d, labels_2d, scores_2d, img_shape, _ = rgb_results[0]  # _ = masks (ignored)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud, collate_data = self.lidar_branch_inference(pc_file)
        
        # Visualize 2D detections
        image = np.array(Image.open(img_file))
        save_path_2d = save_path / 'rgb_detections.png'
        draw_bboxes_2d(image, bboxes_2d.clone().cpu().numpy(), labels_2d.clone().cpu().numpy(), self.class_dict,
                       scores_2d.clone().cpu().numpy(), save_path_2d, self.color_dict,
                       fill=self.visualization_cfg.get('fill_bboxes_2d', True), 
                       alpha=self.visualization_cfg.get('alpha', 50))
        
        # Visualize 3D detections
        bboxes_3d_mmdet = bboxes_3d.clone().cpu()
        bboxes_3d_mmdet[:, 2] -= bboxes_3d_mmdet[:, 5] / 2
        save_path_3d = save_path / 'lidar_detections_proj.png'
        draw_bboxes_3d_image(image, bboxes_3d_mmdet, labels_3d.clone().cpu(), cam_to_img.clone().cpu(), 
                             lidar_to_cam.clone().cpu(), self.color_dict, True, save_path_3d)
        
        # Run final prediction
        final_detections = self.predict(img_file, pc_file, lidar_to_cam, cam_to_img)
        
        # Visualize fusion results
        save_path_fusion = save_path / 'fusion_detections_proj.png'
        draw_bboxes_3d_image(image, final_detections.bboxes_3d.clone().cpu(), 
                             final_detections.labels_3d.clone().cpu(), cam_to_img.clone().cpu(), 
                             lidar_to_cam.clone().cpu(), self.color_dict, True, save_path_fusion)
        
        return final_detections
    
    def set_matching_mode(self, use_clustering: bool):
        """
        Switch between matching modes at runtime.
        
        Args:
            use_clustering (bool): If True, use clustering-based matching. If False, use frustum-based matching.
        """
        self.use_clustering = use_clustering
        if use_clustering:
            print("Switched to clustering-based matching mode")
        else:
            print("Switched to frustum-based matching mode")
