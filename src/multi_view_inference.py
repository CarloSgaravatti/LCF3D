from pathlib import Path
from typing_extensions import Tuple, List, Dict, Union, Sequence
from PIL import Image
import torch
from torch import Tensor
import json
import numpy as np
import warnings
from bbox_utils import *
from visualization.visualization_utils import *
from base_inference import BaseLateFusionInferencer

from mmdet.models.task_modules import BboxOverlaps2D
from mmcv.ops import box_iou_rotated
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import box3d_multiclass_nms, nms_bev
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import networkx as nx


class LateFusionMultiViewInferencer(BaseLateFusionInferencer):
    """
    Late Fusion inference class for multi-view setup (N cameras without overlap).
    
    This class processes multiple camera views independently and aggregates results.
    For each camera view, it performs:
    - Bounding box matching between 3D and 2D detections
    - Semantic fusion (optional)
    - Detection recovery via frustum RPN (optional)
    
    Args:
        late_fusion_cfg (Union[Dict, str, Path]): Configuration dict for late fusion.
            If a file path is passed, it will load the configuration from it
        cam_to_img (List[Union[np.ndarray, Tensor, Sequence[Sequence[float]]]], optional): 
            List of transformation matrices from camera to image coordinates for each view.
        lidar_to_cam (List[Union[np.ndarray, Tensor, Sequence[Sequence[float]]]], optional): 
            List of transformation matrices from LiDAR to camera coordinates for each view.
        device (Union[str, torch.device], optional): Device to run the models on. Defaults to 'cuda:0'.
    """
    def __init__(self,
                 late_fusion_cfg: Union[Dict, str, Path],
                 cam_to_img: List[Union[np.ndarray, Tensor, Sequence[Sequence[float]]]] = None,
                 lidar_to_cam: List[Union[np.ndarray, Tensor, Sequence[Sequence[float]]]] = None,
                 device: Union[str, torch.device] = 'cuda:0'):
        
        # Call parent constructor
        super().__init__(late_fusion_cfg, device)
        
        self.num_views = self.late_fusion_cfg.get('num_views', 1)
        
        # Handle transformation matrices for multiple views
        self.lidar_to_cam = []
        if lidar_to_cam is None:
            warnings.warn('lidar_to_cam is None, if not passed sample by sample the results will be not consistent')
            self.lidar_to_cam = [torch.eye(4, dtype=torch.float32).to(self.device) for _ in range(self.num_views)]
        else:
            for ltc in lidar_to_cam:
                if not isinstance(ltc, Tensor):
                    ltc = torch.tensor(ltc, dtype=torch.float32)
                self.lidar_to_cam.append(ltc.to(self.device))
        
        self.cam_to_img = []
        if cam_to_img is None:
            warnings.warn('cam_to_img is None, if not passed sample by sample the results will be not consistent')
            self.cam_to_img = [torch.eye(4, dtype=torch.float32)[:3, :].to(self.device) for _ in range(self.num_views)]
        else:
            for cti in cam_to_img:
                if not isinstance(cti, Tensor):
                    cti = torch.tensor(cti, dtype=torch.float32)
                if cti.shape[0] == 4:
                    cti = cti[:3, :]
                self.cam_to_img.append(cti.to(self.device))
        
        # Additional multi-view specific configs
        self.num_classes = self.late_fusion_cfg.get('num_classes', 3)
        self.classes = self.late_fusion_cfg.get('classes', [''] * self.num_classes)
        self.class_dict = {i: self.classes[i] for i in range(self.num_classes)}
        
        self.bb_match_iou_thr = self.late_fusion_cfg.get('bbox_matching_iou_thr', 0.5)
        self.recovery_iou_thr = self.late_fusion_cfg.get('detection_recovery_iou_thr', 0.4)
        self.bb_match_mode = self.late_fusion_cfg.get('bbox_matching_mode', 'iou')
        self.min_pts_frustum = self.late_fusion_cfg.get('min_pts_frustum', 10)
        self.truncation_factor_thr = self.late_fusion_cfg.get('truncation_factor_thr', 0.7)
        
        self.use_score_fusion = self.late_fusion_cfg.get('use_score_fusion', True)
        self.class_priors = self.late_fusion_cfg.get('class_prior', [1 / self.num_classes for _ in range(self.num_classes)])
        self.class_priors = torch.tensor(self.class_priors, dtype=torch.float32, device=self.device)
        self.use_final_nms = self.late_fusion_cfg.get('use_final_nms', True)
        self.keep_oov_bboxes = self.late_fusion_cfg.get('keep_oov_bboxes', False)
        self.final_nms_cfg = self.late_fusion_cfg.get('final_nms_cfg', {})
        self.match_class = self.late_fusion_cfg.get('match_class', False)
        
        # Clustering configuration
        self.use_clustering = self.late_fusion_cfg.get('use_clustering', False)
        self.clustering_method = self.late_fusion_cfg.get('clustering_method', 'connected_components')
        if 'use_cliques' in self.late_fusion_cfg:
            self.clustering_method = 'cliques' if self.late_fusion_cfg.get('use_cliques', False) else 'connected_components'

        self.cluster_iou_thr_cc = self.late_fusion_cfg.get('cluster_iou_thr_cc', self.late_fusion_cfg.get('cluster_iou_thr', 0.1))
        self.cluster_iou_thr_clique = self.late_fusion_cfg.get('cluster_iou_thr_clique', self.late_fusion_cfg.get('cluster_iou_thr', 0.1))
        self.cluster_bev_iou_thr = self.late_fusion_cfg.get('cluster_bev_iou_thr', 0.5)
        self.oov_score_thr = self.late_fusion_cfg.get('oov_score_thr', 0.3)
    
    def find_cluster_max(self, bboxes_3d, labels_3d, scores_3d, clusters, num_clusters):
        """
        Find the highest-scoring detection within each cluster.
        
        Args:
            bboxes_3d: Tensor of shape (N, 7) - 3D bounding boxes
            labels_3d: Tensor of shape (N,) - class labels
            scores_3d: Tensor of shape (N,) - detection scores
            clusters: Tensor of shape (N,) - cluster IDs for each detection
            num_clusters: Number of unique clusters
            
        Returns:
            Tuple of (bboxes_3d_max, labels_3d_max, scores_3d_max, max_indices)
        """
        cluster_sortidx = torch.argsort(clusters)
        cluster_ids, cluster_counts = torch.unique_consecutive(clusters[cluster_sortidx], return_counts=True)

        end_indices = torch.cumsum(cluster_counts, dim=0).cpu().tolist()
        start_indices = [0] + end_indices[:-1]

        max_indices = scores_3d.new_zeros((num_clusters,), dtype=torch.long)
        for cluster_id, a, b in zip(cluster_ids, start_indices, end_indices):
            indices = cluster_sortidx[a:b]
            max_indices[cluster_id] = indices[torch.argmax(scores_3d[indices], dim=0)]
            
        bboxes_3d_max = bboxes_3d[max_indices]
        labels_3d_max = labels_3d[max_indices]
        scores_3d_max = scores_3d[max_indices]
        return (
            bboxes_3d_max,
            labels_3d_max,
            scores_3d_max,
            max_indices
        )
    
    def _pre_cluster_3d_bev(self, bboxes_3d, labels_3d, scores_3d):
        """Cluster 3D boxes in BEV space once globally before any per-view processing.

        Groups boxes that represent the same physical object (overlapping in 3D BEV),
        respecting class boundaries. Mirrors the old code's clique/CC clustering that was
        done on 3D BEV IoU rather than on 2D projected IoU.

        Returns:
            cluster_ids: (N_exp,) cluster ID per (possibly expanded) box
            num_clusters: int
            orig_ids: (N_exp,) index into the original bboxes_3d for each expanded slot
        """
        N = bboxes_3d.shape[0]
        if N == 0:
            empty = torch.zeros(0, dtype=torch.long, device=bboxes_3d.device)
            return empty, 0, empty

        bev = self.box_type_3d(bboxes_3d, box_dim=bboxes_3d.shape[-1]).bev
        iou_bev = box_iou_rotated(bev, bev, aligned=False, mode='iou')
        diff_classes = labels_3d[:, None] != labels_3d[None, :]
        iou_bev[diff_classes] = 0

        adj_np = (iou_bev > self.cluster_bev_iou_thr).cpu().numpy()
        np.fill_diagonal(adj_np, 0)

        if self.clustering_method == 'cliques':
            G = nx.from_scipy_sparse_array(sp.csr_matrix(adj_np))
            cliques = list(nx.find_cliques(G))
            cluster_labels = sum([[i] * len(c) for i, c in enumerate(cliques)], [])
            bboxes_ids = sum(cliques, [])
            cluster_ids = torch.tensor(cluster_labels, device=bboxes_3d.device, dtype=torch.long)
            orig_ids = torch.tensor(bboxes_ids, device=bboxes_3d.device, dtype=torch.long)
            num_clusters = len(cliques)
        else:  # connected_components
            num_clusters, cluster_labels = connected_components(
                sp.csr_matrix(adj_np), directed=False)
            cluster_ids = torch.tensor(cluster_labels, device=bboxes_3d.device, dtype=torch.long)
            orig_ids = torch.arange(N, device=bboxes_3d.device)

        return cluster_ids, num_clusters, orig_ids

    def single_view_matching(self, bboxes_proj, cluster_ids, bboxes_2d, num_clusters,
                             match_iou_thr=0.4, mode='iou'):
        """
        Match clusters of 3D boxes to 2D detections using cluster-wise IoU.
        
        Args:
            bboxes_proj: Tensor of shape (N, 4) - projected 3D bboxes in image
            cluster_ids: Tensor of shape (N,) - cluster ID for each projected bbox
            bboxes_2d: Tensor of shape (M, 4) - 2D detections
            num_clusters: Number of clusters
            match_iou_thr: IoU threshold for valid matches
            mode: Mode for IoU calculation ('iou' or other modes from BboxOverlaps2D)
            
        Returns:
            cluster_matching: Tensor of shape (num_clusters,) with matched 2D indices (+1, 0 if no match)
        """
        iou_matrix = BboxOverlaps2D()(bboxes_proj, bboxes_2d, mode=mode)
        reduced_iou = iou_matrix.new_zeros((num_clusters, bboxes_2d.shape[0]))
        reduced_iou = reduced_iou.scatter_reduce_(
            0, cluster_ids.unsqueeze(1).expand(-1, bboxes_2d.shape[0]), 
            iou_matrix, reduce='amax', include_self=False)
        
        lidar_ids, rgb_ids = linear_sum_assignment(-reduced_iou.cpu().numpy())
        lidar_ids = bboxes_proj.new_tensor(lidar_ids, dtype=torch.int)
        rgb_ids = bboxes_proj.new_tensor(rgb_ids, dtype=torch.int)
        assigned_ious = reduced_iou[lidar_ids, rgb_ids]
        valid_matching = assigned_ious > match_iou_thr
        lidar_ids = lidar_ids[valid_matching]
        rgb_ids = rgb_ids[valid_matching]
        cluster_matching = bboxes_proj.new_zeros(size=(num_clusters,), dtype=torch.int)
        cluster_matching[lidar_ids] = rgb_ids + 1
        return cluster_matching
    
    def predict(self, img_files: List[str], pc_file: str,
                lidar_to_cam: List[Tensor] = None, cam_to_img: List[Tensor] = None,
                points: np.ndarray = None):
        """
        Runs inference on the provided images and LiDAR files and returns the final detections.

        Args:
            img_files (List[str]): List of paths to image files (one per view).
            pc_file (str): Path to the LiDAR file.
            lidar_to_cam (List[Tensor], optional): List of transformation matrices from LiDAR to camera coordinates.
            cam_to_img (List[Tensor], optional): List of transformation matrices from camera to image coordinates.

        Returns:
            InstanceData: The final detections including 3D bounding boxes, scores, and labels.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img is None:
            cam_to_img = self.cam_to_img
        
        assert len(img_files) == len(lidar_to_cam) == len(cam_to_img), \
            f"Number of images ({len(img_files)}) must match number of transformations"
        
        # Run branch inferences
        rgb_results = self.rgb_branch_inference(img_files)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud, collate_data = self.lidar_branch_inference(
            pc_file, points=points
        )
        
        # Pre-cluster 3D boxes in BEV space (once globally, before any view loop).
        # Old code did this on 3D BEV IoU with class awareness; the 2D projected clustering
        # that was here before is NOT equivalent and produced worse results.
        if self.use_clustering:
            cluster_ids, num_clusters, orig_ids = self._pre_cluster_3d_bev(
                bboxes_3d, labels_3d, scores_3d)
            bboxes_3d = bboxes_3d[orig_ids]
            corners_3d = corners_3d[orig_ids]
            labels_3d = labels_3d[orig_ids]
            scores_3d = scores_3d[orig_ids]
        else:
            cluster_ids = None
            num_clusters = None

        # Only cluster representatives (max-score per cluster) can be OOV candidates.
        # Non-representative cluster members are not returned as standalone detections —
        # they exist only to improve cluster IoU matching. Mirrors old code:
        #   out_of_view_mask[max_indices] = True  (everything else starts False)
        oov_mask = torch.zeros(bboxes_3d.shape[0], dtype=torch.bool, device=self.device)
        if self.use_clustering and cluster_ids is not None:
            _, _, _, rep_indices = self.find_cluster_max(
                bboxes_3d, labels_3d, scores_3d, cluster_ids, num_clusters)
            oov_mask[rep_indices] = True
        else:
            oov_mask[:] = True

        # Store matched detections from all views
        all_matched_bboxes_3d = []
        all_matched_scores_3d = []
        all_matched_labels_3d = []
        all_matched_bboxes_2d = []
        all_matched_scores_2d = []
        all_matched_labels_2d = []
        
        # Store unmatched 2D detections for recovery
        all_unmatched_bboxes_2d = []
        all_unmatched_scores_2d = []
        all_unmatched_labels_2d = []
        all_unmatched_masks_2d = []  # NEW: Store masks for recovery
        all_unmatched_img_shapes = []
        all_unmatched_lidar_to_cam = []
        all_unmatched_cam_to_img = []
        
        # Process each view independently
        for view_idx, (bboxes_2d, labels_2d, scores_2d, img_shape, masks_2d) in enumerate(rgb_results):
            ltc = lidar_to_cam[view_idx]
            cti = cam_to_img[view_idx]
            
            # Perform bbox matching for this view
            matching_result = self.bbox_matching_single_view(
                bboxes_3d, scores_3d, labels_3d, corners_3d,
                bboxes_2d, scores_2d, labels_2d, img_shape,
                ltc, cti, cluster_ids=cluster_ids, num_clusters=num_clusters
            )
            
            # Update OOV mask (intersection across all views)
            oov_mask = oov_mask & matching_result['oov_mask']
            
            # Collect matched detections
            if matching_result['matched_bboxes_3d'].shape[0] > 0:
                all_matched_bboxes_3d.append(matching_result['matched_bboxes_3d'])
                all_matched_scores_3d.append(matching_result['matched_scores_3d'])
                all_matched_labels_3d.append(matching_result['matched_labels_3d'])
                
                if self.use_label_fusion:
                    all_matched_bboxes_2d.append(matching_result['matched_bboxes_2d'])
                    all_matched_scores_2d.append(matching_result['matched_scores_2d'])
                    all_matched_labels_2d.append(matching_result['matched_labels_2d'])
            
            # Collect unmatched 2D detections for recovery
            if self.use_detection_recovery and matching_result['unmatched_bboxes_2d'].shape[0] > 0:
                all_unmatched_bboxes_2d.append(matching_result['unmatched_bboxes_2d'])
                all_unmatched_scores_2d.append(matching_result['unmatched_scores_2d'])
                all_unmatched_labels_2d.append(matching_result['unmatched_labels_2d'])
                all_unmatched_masks_2d.append(masks_2d)  # NEW: Pass masks for this view
                all_unmatched_img_shapes.append(img_shape)
                all_unmatched_lidar_to_cam.append(ltc)
                all_unmatched_cam_to_img.append(cti)
        
        # Aggregate matched detections
        if len(all_matched_bboxes_3d) > 0:
            matched_bboxes_3d = torch.cat(all_matched_bboxes_3d, dim=0)
            matched_scores_3d = torch.cat(all_matched_scores_3d, dim=0)
            matched_labels_3d = torch.cat(all_matched_labels_3d, dim=0)
            
            # Apply semantic fusion if enabled
            if self.use_label_fusion and len(all_matched_bboxes_2d) > 0:
                matched_bboxes_2d = torch.cat(all_matched_bboxes_2d, dim=0)
                matched_scores_2d = torch.cat(all_matched_scores_2d, dim=0)
                matched_labels_2d = torch.cat(all_matched_labels_2d, dim=0)
                
                matched_labels_3d, matched_scores_3d = self.semantic_fusion(
                    matched_bboxes_3d, matched_scores_3d, matched_labels_3d,
                    matched_bboxes_2d, matched_scores_2d, matched_labels_2d
                )
        else:
            matched_bboxes_3d = bboxes_3d.new_empty((0, bboxes_3d.shape[1]))
            matched_scores_3d = scores_3d.new_empty((0,))
            matched_labels_3d = labels_3d.new_empty((0,), dtype=torch.long)
        
        # Detection recovery for unmatched 2D boxes
        if self.use_detection_recovery and self.frustum_detector is not None and len(all_unmatched_bboxes_2d) > 0:
            recovered_bboxes_3d, recovered_scores_3d, recovered_labels_3d = self.detection_recovery_multi_view(
                collate_data, all_unmatched_bboxes_2d, all_unmatched_scores_2d, all_unmatched_labels_2d,
                all_unmatched_lidar_to_cam, all_unmatched_cam_to_img, all_unmatched_img_shapes,
                unmatched_masks_2d_list=all_unmatched_masks_2d  # NEW: Pass masks
            )
            
            if recovered_bboxes_3d.shape[0] > 0:
                # Pad recovered boxes if 3D detector outputs more dims (e.g. 9D with velocity vs 7D frustum output)
                if recovered_bboxes_3d.shape[1] < matched_bboxes_3d.shape[1]:
                    pad = matched_bboxes_3d.shape[1] - recovered_bboxes_3d.shape[1]
                    recovered_bboxes_3d = torch.cat(
                        [recovered_bboxes_3d, recovered_bboxes_3d.new_zeros((recovered_bboxes_3d.shape[0], pad))], dim=1
                    )
                matched_bboxes_3d = torch.cat([matched_bboxes_3d, recovered_bboxes_3d], dim=0)
                matched_scores_3d = torch.cat([matched_scores_3d, recovered_scores_3d], dim=0)
                matched_labels_3d = torch.cat([matched_labels_3d, recovered_labels_3d], dim=0)
        
        # Add out-of-view detections if enabled (with score threshold filtering)
        if self.keep_oov_bboxes:
            oov_bboxes_3d = bboxes_3d[oov_mask]
            oov_scores_3d = scores_3d[oov_mask]
            oov_labels_3d = labels_3d[oov_mask]
            
            # Filter OOV boxes by score threshold
            valid_oov = oov_scores_3d >= self.oov_score_thr
            oov_bboxes_3d = oov_bboxes_3d[valid_oov]
            oov_scores_3d = oov_scores_3d[valid_oov]
            oov_labels_3d = oov_labels_3d[valid_oov]
            
            if oov_bboxes_3d.shape[0] > 0:
                matched_bboxes_3d = torch.cat([matched_bboxes_3d, oov_bboxes_3d], dim=0)
                matched_scores_3d = torch.cat([matched_scores_3d, oov_scores_3d], dim=0)
                matched_labels_3d = torch.cat([matched_labels_3d, oov_labels_3d], dim=0)
        
        # Final NMS
        if self.use_final_nms and matched_bboxes_3d.shape[0] > 0:
            nms_cfg = ConfigDict(use_rotate_nms=True, nms_thr=self.final_nms_cfg.get('thresh', 0.3))
            score_thr = self.final_nms_cfg.get('score_thr', 0.01)
            box_dim = matched_bboxes_3d.shape[-1]
            bev_boxes_for_nms = xywhr2xyxyr(self.box_type_3d(matched_bboxes_3d, box_dim=box_dim).bev)
            scores_for_nms = matched_bboxes_3d.new_zeros((matched_scores_3d.shape[0], self.num_classes), dtype=torch.float32)
            scores_for_nms[torch.arange(matched_scores_3d.shape[0], dtype=torch.long), matched_labels_3d.long()] = matched_scores_3d
            # Separate velocity attributes (cols 7+) before NMS, re-attach after (mirrors old code behaviour)
            attrs = matched_bboxes_3d[:, 7:] if box_dim > 7 else None
            boxes_7d = matched_bboxes_3d[:, :7]
            nms_results = box3d_multiclass_nms(
                boxes_7d, bev_boxes_for_nms, scores_for_nms,
                score_thr=score_thr, max_num=100, cfg=nms_cfg,
                mlvl_attr_scores=attrs)
            if attrs is None:
                matched_bboxes_3d, matched_scores_3d, matched_labels_3d = nms_results
            else:
                matched_bboxes_3d, matched_scores_3d, matched_labels_3d, attrs = nms_results
                matched_bboxes_3d = torch.cat([matched_bboxes_3d, attrs], dim=-1)
        
        final_detections = InstanceData()
        final_detections.bboxes_3d = self.box_type_3d(matched_bboxes_3d, box_dim=matched_bboxes_3d.shape[-1])
        final_detections.scores_3d = matched_scores_3d
        final_detections.labels_3d = matched_labels_3d
        return final_detections
    
    def visualize_predict(self, img_files: List[str], pc_file: str, save_path: Union[str, Path],
                         lidar_to_cam: List[Tensor] = None, cam_to_img: List[Tensor] = None):
        """
        Runs inference and saves visualizations for each view.

        Args:
            img_files (List[str]): List of paths to image files.
            pc_file (str): Path to the LiDAR file.
            save_path (Union[str, Path]): Directory to save visualization images.
            lidar_to_cam (List[Tensor], optional): List of transformation matrices.
            cam_to_img (List[Tensor], optional): List of transformation matrices.

        Returns:
            InstanceData: The final detections.
        """
        if lidar_to_cam is None:
            lidar_to_cam = self.lidar_to_cam
        if cam_to_img is None:
            cam_to_img = self.cam_to_img
        if isinstance(save_path, str):
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Run branch inferences
        rgb_results = self.rgb_branch_inference(img_files)
        bboxes_3d, corners_3d, labels_3d, scores_3d, point_cloud, collate_data = self.lidar_branch_inference(pc_file)
        
        # Visualize 2D and 3D detections for each view
        for view_idx, (bboxes_2d, labels_2d, scores_2d, img_shape, masks_2d) in enumerate(rgb_results):
            image = np.array(Image.open(img_files[view_idx]))
            
            # Visualize 2D detections
            save_path_2d = save_path / f'rgb_detections_view{view_idx}.png'
            draw_bboxes_2d(
                image, bboxes_2d.clone().cpu().numpy(), labels_2d.clone().cpu().numpy(), 
                self.class_dict, scores_2d.clone().cpu().numpy(), save_path_2d, self.color_dict,
                fill=self.visualization_cfg.get('fill_bboxes_2d', True),
                alpha=self.visualization_cfg.get('alpha', 50)
            )
            
            # Visualize 3D detections projected
            bboxes_3d_mmdet = bboxes_3d.clone().cpu()
            bboxes_3d_mmdet[:, 2] -= bboxes_3d_mmdet[:, 5] / 2
            save_path_3d = save_path / f'lidar_detections_view{view_idx}_proj.png'
            draw_bboxes_3d_image(
                image, bboxes_3d_mmdet, labels_3d.clone().cpu(),
                cam_to_img[view_idx].clone().cpu(), lidar_to_cam[view_idx].clone().cpu(),
                self.color_dict, True, save_path_3d
            )
        
        # Run final prediction
        final_detections = self.predict(img_files, pc_file, lidar_to_cam, cam_to_img)
        
        # Visualize fusion results for each view
        for view_idx in range(len(img_files)):
            image = np.array(Image.open(img_files[view_idx]))
            save_path_fusion = save_path / f'fusion_detections_view{view_idx}_proj.png'
            draw_bboxes_3d_image(
                image, final_detections.bboxes_3d.clone().cpu(),
                final_detections.labels_3d.clone().cpu(),
                cam_to_img[view_idx].clone().cpu(), lidar_to_cam[view_idx].clone().cpu(),
                self.color_dict, True, save_path_fusion
            )
        
        return final_detections
    
    def bbox_matching_single_view(self, bboxes_3d, scores_3d, labels_3d, corners_3d,
                                  bboxes_2d, scores_2d, labels_2d, img_shape,
                                  lidar_to_cam, cam_to_img,
                                  cluster_ids=None, num_clusters=None):
        """
        Match 3D bboxes with 2D detections for a single view.
        
        Returns:
            Dict with:
            - matched_bboxes_3d, matched_scores_3d, matched_labels_3d
            - matched_bboxes_2d, matched_scores_2d, matched_labels_2d (for semantic fusion)
            - unmatched_bboxes_2d, unmatched_scores_2d, unmatched_labels_2d (for recovery)
            - oov_mask: boolean mask for out-of-view 3D detections
        """
        # Check if 3D boxes are in field of view
        centers = torch.cat([bboxes_3d[:, :3], torch.ones_like(bboxes_3d[:, :1])], dim=-1)
        in_field_of_view = lidar_to_cam.matmul(centers.t()).t()
        in_field_of_view = in_field_of_view[:, 2] > 0
        
        # Project 3D bboxes to image
        bboxes_proj = corners_to_axis_aligned_bbox(corners_3d[in_field_of_view], P=cam_to_img, lidar=True, T=lidar_to_cam)
        
        # Calculate truncation factor
        width_pre = torch.abs(bboxes_proj[:, 2] - bboxes_proj[:, 0])
        bboxes_proj = clamp_boxes(bboxes_proj, img_shape)
        truncation_factor = (bboxes_proj[:, 2] - bboxes_proj[:, 0]) / (width_pre + 1e-6)
        
        # Filter boxes inside image with acceptable truncation
        inside_image = (bboxes_proj[:, 0] < bboxes_proj[:, 2]) & \
                      (bboxes_proj[:, 1] < bboxes_proj[:, 3]) & \
                      (truncation_factor > self.truncation_factor_thr)
        
        # Update in_field_of_view mask
        valid_mask = in_field_of_view.clone()
        valid_mask[in_field_of_view] = inside_image
        
        # Get valid 3D boxes
        bboxes_3d_valid = bboxes_3d[valid_mask]
        scores_3d_valid = scores_3d[valid_mask]
        labels_3d_valid = labels_3d[valid_mask]
        bboxes_proj_valid = bboxes_proj[inside_image]
        
        # Initialize result dict
        # oov_mask starts all-True; only matched boxes will be set to False below.
        # This mirrors old code: a box is "not OOV" only if matched to a 2D detection,
        # so visible-but-unmatched boxes are kept just like geometrically-OOV boxes.
        result = {
            'oov_mask': torch.ones(bboxes_3d.shape[0], dtype=torch.bool, device=bboxes_3d.device),
            'matched_bboxes_3d': bboxes_3d.new_empty((0, bboxes_3d.shape[1])),
            'matched_scores_3d': scores_3d.new_empty((0,)),
            'matched_labels_3d': labels_3d.new_empty((0,), dtype=torch.long),
            'matched_bboxes_2d': bboxes_2d.new_empty((0, 4)),
            'matched_scores_2d': scores_2d.new_empty((0,)),
            'matched_labels_2d': labels_2d.new_empty((0,), dtype=torch.long),
            'unmatched_bboxes_2d': bboxes_2d,
            'unmatched_scores_2d': scores_2d,
            'unmatched_labels_2d': labels_2d,
        }
        
        if bboxes_3d_valid.shape[0] == 0 or bboxes_2d.shape[0] == 0:
            return result
        
        # Perform matching with optional clustering
        if self.use_clustering and cluster_ids is not None:
            # Use pre-computed 3D BEV cluster IDs (computed once globally in predict()).
            # Filter to boxes visible in this view and re-map cluster IDs to 0..N-1.
            cluster_ids_valid = cluster_ids[valid_mask]
            unique_clusters = torch.unique(cluster_ids_valid)
            num_visible_clusters = unique_clusters.shape[0]
            cluster_remap = cluster_ids_valid.new_full(
                (int(cluster_ids_valid.max().item()) + 1,), -1)
            cluster_remap[unique_clusters] = torch.arange(
                num_visible_clusters, device=cluster_ids_valid.device,
                dtype=cluster_ids_valid.dtype)
            cluster_ids_local = cluster_remap[cluster_ids_valid]

            # Reduce IoU per cluster (max across all boxes in cluster), Hungarian match
            cluster_matching = self.single_view_matching(
                bboxes_proj_valid, cluster_ids_local,
                bboxes_2d, num_visible_clusters,
                match_iou_thr=self.bb_match_iou_thr,
                mode=self.bb_match_mode
            )

            matches_mask = cluster_matching > 0
            matched_2d_indices = (cluster_matching - 1)[matches_mask].long()
            matched_cluster_indices = torch.nonzero(matches_mask).squeeze(1)

            # Representative of each matched cluster = max-score box in that cluster
            _, _, _, max_indices_local = self.find_cluster_max(
                bboxes_3d_valid, labels_3d_valid, scores_3d_valid,
                cluster_ids_local, num_visible_clusters
            )
            matched_3d_mask = torch.zeros(
                bboxes_3d_valid.shape[0], dtype=torch.bool, device=bboxes_3d_valid.device)
            matched_3d_mask[max_indices_local[matched_cluster_indices]] = True
        else:
            # Standard matching without clustering
            if self.match_class:
                assignment = match_bboxes_linear_sum_assign_cls(
                    bboxes_proj_valid, bboxes_2d, labels_3d_valid, labels_2d,
                    scores_3d_valid, scores_2d, mode=self.bb_match_mode,
                    iou_thr=self.bb_match_iou_thr, conf_lambda=None
                )
            else:
                assignment = match_bboxes_linear_sum_assign(
                    self.iou_calculator, bboxes_proj_valid, bboxes_2d,
                    mode=self.bb_match_mode, iou_thr=self.bb_match_iou_thr
                )
            
            # Get matched indices
            matches_mask = assignment >= 1
            matched_2d_indices = torch.clamp(assignment - 1, min=-1)[matches_mask].long()
            matched_3d_mask = matches_mask
        
        # Get unmatched 2D boxes
        unmatched_2d_mask = torch.isin(
            torch.arange(bboxes_2d.shape[0]).to(bboxes_2d.device),
            matched_2d_indices, invert=True
        )
        
        # Update result
        result['matched_bboxes_3d'] = bboxes_3d_valid[matched_3d_mask]
        result['matched_scores_3d'] = scores_3d_valid[matched_3d_mask]
        result['matched_labels_3d'] = labels_3d_valid[matched_3d_mask]
        result['matched_bboxes_2d'] = bboxes_2d[matched_2d_indices]
        result['matched_scores_2d'] = scores_2d[matched_2d_indices]
        result['matched_labels_2d'] = labels_2d[matched_2d_indices]
        result['unmatched_bboxes_2d'] = bboxes_2d[unmatched_2d_mask]
        result['unmatched_scores_2d'] = scores_2d[unmatched_2d_mask]
        result['unmatched_labels_2d'] = labels_2d[unmatched_2d_mask]

        # Mark matched 3D boxes as not-OOV. Visible-but-unmatched boxes stay True
        # (oov_mask=True) so they are kept alongside geometrically-OOV boxes — same
        # behaviour as the old code where out_of_view_mask tracked "not matched anywhere".
        valid_indices = torch.where(valid_mask)[0]
        result['oov_mask'][valid_indices[matched_3d_mask]] = False

        return result
    
    def semantic_fusion(self, bboxes_3d, scores_3d, labels_3d, bboxes_2d, scores_2d, labels_2d):
        """
        Fuse labels and scores from 2D and 3D detections.

        When labels agree: score = score_3d * score_2d / class_prior (Bayesian product, unnormalized)
        When labels disagree: score = score_2d, label = label_2d
        Always returns 2D label (semantic fusion mode).

        Mirrors old code in multi_view_single_frustum_with_cluster.py::semantic_fusion().
        """
        different_labels = labels_3d != labels_2d
        new_scores = scores_3d.clone()
        new_labels = labels_2d.clone()

        # Agreed labels: product / prior (unnormalized Bayesian)
        agree = ~different_labels
        new_scores[agree] = (scores_3d[agree] * scores_2d[agree]) / (
            self.class_priors[labels_2d[agree].long()] + 1e-8)

        # Disagreed labels: fall back to 2D score
        new_scores[different_labels] = scores_2d[different_labels]

        return new_labels, new_scores
    
    def detection_recovery_multi_view(self, collate_data, unmatched_bboxes_2d_list,
                                     unmatched_scores_2d_list, unmatched_labels_2d_list,
                                     lidar_to_cam_list, cam_to_img_list, img_shape_list,
                                     unmatched_masks_2d_list=None):
        """
        Recovery detections using frustum RPN for all unmatched 2D boxes across all views.
        
        Args:
            unmatched_masks_2d_list: Optional list of instance masks for 2D detections (for mask-aware filtering)
        """
        scan = collate_data['inputs']['points'][0].to(self.device)
        scan = scan[scan[:, 0] > 0]

        scan_for_frustum = scan
        if getattr(self, 'use_dims_frustum', None) is not None:
            if len(self.use_dims_frustum) < 3 or self.use_dims_frustum[:3] != [0, 1, 2]:
                raise ValueError('use_dims_frustum must start with [0, 1, 2] so frustum_pc[:, :3] is xyz.')
            scan_for_frustum = scan[:, self.use_dims_frustum]

        points = scan[:, :3]
        
        all_recovered_bboxes = []
        all_recovered_scores = []
        all_recovered_labels = []
        
        # Process each view
        for view_idx, unmatched_bboxes_2d in enumerate(unmatched_bboxes_2d_list):
            if unmatched_bboxes_2d.shape[0] == 0:
                continue
            
            lidar_to_cam = lidar_to_cam_list[view_idx]
            cam_to_img = cam_to_img_list[view_idx]
            unmatched_scores_2d = unmatched_scores_2d_list[view_idx]
            unmatched_labels_2d = unmatched_labels_2d_list[view_idx]
            img_shape = img_shape_list[view_idx]
            
            # Project points to image
            velo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            coord = cam_to_img.matmul(lidar_to_cam.matmul(velo.t()))
            coord[:2] /= coord[2, :]
            coord[2] = 1
            coord = coord.T
            
            # Camera coordinates for depth check
            cam_coords = lidar_to_cam.matmul(velo.t())
            
            # Enlarge 2D boxes
            bboxes_2d_enlarge = enlarge_bboxes_2d(unmatched_bboxes_2d.clone(), self.enlarge_factor, self.enlarge_factor)
            
            # Find points in each frustum
            fov_inds = (
                (coord[:, 0, None] <= bboxes_2d_enlarge[:, 2]) &
                (coord[:, 0, None] >= bboxes_2d_enlarge[:, 0]) &
                (coord[:, 1, None] <= bboxes_2d_enlarge[:, 3]) &
                (coord[:, 1, None] >= bboxes_2d_enlarge[:, 1]) &
                (cam_coords[2, :, None] > 0)
            )
            
            num_points = fov_inds.sum(dim=0).int()
            valid_indices = torch.nonzero(num_points > self.min_pts_frustum).squeeze(1)

            # Get masks for this view if available
            masks_2d = None
            if unmatched_masks_2d_list is not None and unmatched_masks_2d_list[view_idx] is not None:
                masks_2d = unmatched_masks_2d_list[view_idx][valid_indices]
            
            if valid_indices.shape[0] == 0:
                continue
            
            # Prepare one-hot vectors
            one_hot_vectors = torch.nn.functional.one_hot(
                unmatched_labels_2d, num_classes=self.valid_2d_classes.shape[0]
            )
            
            # Alignment preprocessing
            if self.align_frustum:
                K_inv = torch.linalg.inv(cam_to_img[:, :3])
                cam_to_lidar = torch.linalg.inv(lidar_to_cam)
            
            # Process each frustum
            frustum_bboxes = []
            frustum_scores = []
            frustum_labels = []
            
            for i, idx in enumerate(valid_indices):
                fov_mask = fov_inds[:, idx]
                frustum_pc = scan_for_frustum[fov_mask, :].clone()
                
                # Apply mask filtering if masks are available
                if masks_2d is not None and masks_2d[i] is not None:
                    # Filter points based on mask
                    valid_mask_points = masks_2d[i][coord[fov_mask, 0].long(), coord[fov_mask, 1].long()]
                    frustum_pc = frustum_pc[valid_mask_points]
                
                if frustum_pc.shape[0] == 0:
                    continue
                
                # Add Gaussian likelihoods
                if self.use_gaussian_likelihoods:
                    ori_box = unmatched_bboxes_2d[idx]
                    wl = ori_box[2] - ori_box[0]
                    hl = ori_box[3] - ori_box[1]
                    xl = ori_box[0] + wl / 2
                    yl = ori_box[1] + hl / 2
                    likelihoods = torch.exp(
                        -((coord[fov_mask, 0] - xl) ** 2 / (2 * wl ** 2)) -
                        ((coord[fov_mask, 1] - yl) ** 2 / (2 * hl ** 2))
                    )
                    frustum_pc = torch.cat([frustum_pc, likelihoods.unsqueeze(1)], dim=-1)
                
                # Align frustum
                if self.align_frustum:
                    box_center = torch.cat([
                        (bboxes_2d_enlarge[idx, 2:4] + bboxes_2d_enlarge[idx, 0:2]) / 2,
                        torch.ones(1, device=self.device)
                    ])
                    backproj = K_inv.matmul(box_center)
                    backproj = torch.cat([backproj, torch.zeros(1, device=self.device)])
                    backproj = cam_to_lidar.matmul(backproj)
                    yaw_lidar = torch.atan2(backproj[1], backproj[0])
                    
                    rotation_matrix = scan.new_tensor([
                        [torch.cos(-yaw_lidar), -torch.sin(-yaw_lidar), 0],
                        [torch.sin(-yaw_lidar), torch.cos(-yaw_lidar), 0],
                        [0, 0, 1],
                    ])
                    frustum_pc[:, :3] = rotation_matrix.matmul(frustum_pc[:, :3].t()).t()
                
                # Prepare data sample (reuse the same object — set_metainfo overwrites one_hot_vector each iter)
                data_sample = collate_data['data_samples'][0]
                data_sample.set_metainfo({'one_hot_vector': one_hot_vectors[idx, :]})
                
                proposal = {
                    'data_samples': [data_sample],
                    'inputs': {'points': [frustum_pc]}
                }
                
                # Run frustum detector
                with torch.no_grad():
                    detection_output = self.frustum_detector.test_step(proposal)
                
                # Parse output
                if len(detection_output) == 4:
                    _, new_bboxes_3d, new_scores_3d, new_labels_3d = detection_output
                elif len(detection_output) == 3:
                    _, new_bboxes_3d, new_scores_3d = detection_output
                    new_labels_3d = unmatched_labels_2d[idx].unsqueeze(0).expand(new_bboxes_3d.shape[0])
                else:
                    _, new_bboxes_3d = detection_output
                    new_scores_3d = unmatched_scores_2d[idx].unsqueeze(0).expand(new_bboxes_3d.shape[0])
                    new_labels_3d = unmatched_labels_2d[idx].unsqueeze(0).expand(new_bboxes_3d.shape[0])
                
                # De-align frustum (rotation_matrix is on self.device; keep new_bboxes_3d on same device)
                if self.align_frustum:
                    new_bboxes_3d[:, :3] = torch.linalg.inv(rotation_matrix).matmul(new_bboxes_3d[:, :3].t()).t()
                    new_bboxes_3d[:, -1] += yaw_lidar

                # Center adjustment (required by project_bboxes which treats z as center)
                new_bboxes_3d[:, 2] += new_bboxes_3d[:, 5] / 2

                # IoU filtering with 2D box — project_bboxes is pure PyTorch, works on any device
                bboxes_proj = project_bboxes(new_bboxes_3d, cam_to_img, lidar=True, T=lidar_to_cam)
                ious = self.iou_calculator(bboxes_proj, unmatched_bboxes_2d[idx:idx+1], mode='iou', is_aligned=False)
                iou_filter = ious.squeeze(1) > self.recovery_iou_thr

                # Shift z back to bottom before storing
                new_bboxes_3d[:, 2] -= new_bboxes_3d[:, 5] / 2

                if iou_filter.sum() > 0:
                    frustum_bboxes.append(new_bboxes_3d[iou_filter])
                    frustum_scores.append(new_scores_3d[iou_filter] if isinstance(new_scores_3d, Tensor) else torch.tensor(new_scores_3d[iou_filter], device=self.device))
                    frustum_labels.append(new_labels_3d[iou_filter] if isinstance(new_labels_3d, Tensor) else torch.tensor(new_labels_3d[iou_filter], device=self.device))
            
            # Aggregate frustum detections for this view
            if len(frustum_bboxes) > 0:
                all_recovered_bboxes.append(torch.cat(frustum_bboxes, dim=0))
                all_recovered_scores.append(torch.cat(frustum_scores, dim=0))
                all_recovered_labels.append(torch.cat(frustum_labels, dim=0))
        
        # Aggregate all recovered detections
        if len(all_recovered_bboxes) > 0:
            recovered_bboxes_3d = torch.cat(all_recovered_bboxes, dim=0).to(self.device)
            recovered_scores_3d = torch.cat(all_recovered_scores, dim=0).to(self.device)
            recovered_labels_3d = torch.cat(all_recovered_labels, dim=0).to(self.device)
        else:
            recovered_bboxes_3d = torch.empty((0, 7), device=self.device)
            recovered_scores_3d = torch.empty((0,), device=self.device)
            recovered_labels_3d = torch.empty((0,), dtype=torch.long, device=self.device)
        
        return recovered_bboxes_3d, recovered_scores_3d, recovered_labels_3d
