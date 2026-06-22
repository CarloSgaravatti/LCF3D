from pathlib import Path
from typing_extensions import Tuple, List, Dict, Union, Sequence
from copy import deepcopy
from PIL import Image
import numpy as np
import torch
from torch import Tensor
import json
from abc import ABC, abstractmethod

from bbox_utils import *
from visualization.visualization_utils import *

from mmdet.models.task_modules import BboxOverlaps2D
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose as ComposeMMCV

from mmdet3d.structures import get_box_type
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.apis.inference import init_model, inference_detector
from mmengine.dataset import Compose as ComposeMMEngine, pseudo_collate
from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import box3d_multiclass_nms, nms_bev
from mmengine.structures import InstanceData
from mmengine.config import Config
from ultralytics import YOLO


class BaseLateFusionInferencer(ABC):
    """
    Base class for Late Fusion inference with common methods.
    
    This class provides the shared functionality for both single-view and stereo-view
    late fusion inference, including 2D/3D detector initialization and branch inference.
    
    Args:
        late_fusion_cfg (Union[Dict, str, Path]): Configuration dict for late fusion.
            If a file path is passed, it will load the configuration from it
        device (Union[str, torch.device], optional): Device to run the models on. Defaults to 'cuda:0'.
    """
    
    def __init__(self,
                 late_fusion_cfg: Union[Dict, str, Path],
                 device: Union[str, torch.device] = 'cuda:0'):
        
        # Load configuration
        if isinstance(late_fusion_cfg, (str, Path)):
            with open(late_fusion_cfg, 'r') as f:
                late_fusion_cfg = json.load(f)
        
        self.device = device
        self.late_fusion_cfg = late_fusion_cfg
        
        # Extract 2D detector configuration
        self.detector2d_cfg = late_fusion_cfg.get('detector2d', {})
        self.model_type = self.detector2d_cfg.get('model_type', 'mmdet')
        self.score_thr_2d = torch.tensor(self.detector2d_cfg.get('score_thr', [0.5] * 10), device=device)
        self.valid_2d_classes = torch.tensor(self.detector2d_cfg.get('valid_2d_classes', list(range(10))), device=device)
        self.img_label_mapping = torch.tensor(self.detector2d_cfg.get('img_label_mapping', list(range(10))), device=device)
        
        # Extract 3D detector configuration
        self.detector3d_cfg = late_fusion_cfg.get('detector3d', {})
        score_thr_3d = self.detector3d_cfg.get('score_thr', late_fusion_cfg.get('score_thr_3d', [0.5] * 10))
        if isinstance(score_thr_3d, (int, float)):
            score_thr_3d = [float(score_thr_3d)] * self.valid_2d_classes.numel()
        self.score_thr_3d = torch.tensor(score_thr_3d, device=device)

        valid_labels_3d = self.detector3d_cfg.get('valid_labels_3d', late_fusion_cfg.get('valid_labels_3d', None))
        if valid_labels_3d is None:
            valid_labels_3d = list(range(int(self.score_thr_3d.numel())))
        self.valid_labels_3d = torch.tensor(valid_labels_3d, device=device).long()

        class_mapping_3d = self.detector3d_cfg.get('class_mapping_3d', late_fusion_cfg.get('class_mapping_3d', None))
        if class_mapping_3d is None:
            class_mapping_3d = list(range(int(self.score_thr_3d.numel())))
        elif isinstance(class_mapping_3d, dict):
            max_key = max([int(k) for k in class_mapping_3d.keys()], default=-1)
            mapping_list = list(range(max_key + 1))
            for k, v in class_mapping_3d.items():
                mapping_list[int(k)] = int(v)
            class_mapping_3d = mapping_list
        self.class_mapping_3d = torch.tensor(class_mapping_3d, device=device).long()
        # NOTE: box_type/mode are derived from the detector cfg pipeline (see _init_detector3d).
        self.box_type_3d = None
        self.box_mode_3d = None
        
        # Extract frustum detector configuration
        self.frustum_detector_cfg = late_fusion_cfg.get('frustum_detector', {})
        
        # Extract fusion configuration (nested section; fall back to flat top-level keys)
        fusion_cfg = late_fusion_cfg.get('fusion', {})
        self.match_iou_thr = fusion_cfg.get('match_iou_thr', 0.1)
        self.use_label_fusion = fusion_cfg.get('use_label_fusion', late_fusion_cfg.get('use_label_fusion', True))
        self.label_fusion_cfg = fusion_cfg.get('label_fusion', {})

        # Detection recovery configuration (nested section; fall back to flat top-level keys)
        recovery_cfg = late_fusion_cfg.get('detection_recovery', {})
        self.use_detection_recovery = recovery_cfg.get('use_detection_recovery', late_fusion_cfg.get('use_detection_recovery', True))
        self.enlarge_factor = recovery_cfg.get('enlarge_factor', late_fusion_cfg.get('enlarge_factor', 0.1))
        self.use_gaussian_likelihoods = recovery_cfg.get('use_gaussian_likelihoods', late_fusion_cfg.get('use_gaussian_likelihoods', True))
        self.align_frustum = recovery_cfg.get('align_frustum', late_fusion_cfg.get('align_frustum', True))
        # If not None, selects a subset of point dimensions when building frustum
        # proposals for the frustum detector (e.g., [0,1,2] or [0,1,2,4]).
        self.use_dims_frustum = recovery_cfg.get('use_dims_frustum', late_fusion_cfg.get('use_dims_frustum', None))
        if self.use_dims_frustum is not None:
            self.use_dims_frustum = [int(i) for i in self.use_dims_frustum]
        
        # Visualization configuration
        self.visualization_cfg = late_fusion_cfg.get('visualization', {})
        self.class_dict = self.visualization_cfg.get('class_dict', {})
        self.color_dict = self.visualization_cfg.get('color_dict', {})
        
        # Initialize detectors
        self._init_detector3d()
        self._init_detector2d()
        
        # Initialize frustum detector if detection recovery is enabled
        if self.use_detection_recovery:
            self._init_frustum_detector()
        else:
            self.frustum_detector = None
        
        # IOU calculator for matching
        self.iou_calculator = BboxOverlaps2D()
    
    def _init_detector3d(self):
        """Initialize the 3D detector."""
        cfg_file = self.detector3d_cfg.get('cfg_path', None)
        checkpoint = self.detector3d_cfg.get('checkpoint_path', None)

        # Optional: override detector test-time NMS parameters from late-fusion cfg.
        nms_thr = self.detector3d_cfg.get('nms_thr', None)
        nms_pre = self.detector3d_cfg.get('nms_pre', None)
        max_num = self.detector3d_cfg.get('max_num', None)

        cfg_for_init = cfg_file
        if cfg_file is not None and any(v is not None for v in [nms_thr, nms_pre, max_num]):
            cfg_for_init = Config.fromfile(cfg_file)
            try:
                model_type = cfg_for_init.model.type
            except Exception:
                model_type = None

            try:
                if model_type in ['PartA2', 'PointVoxelRCNN']:
                    if nms_thr is not None:
                        cfg_for_init.model.test_cfg.rcnn['nms_thr'] = float(nms_thr)
                        if cfg_for_init.model.test_cfg.rpn.get('nms_thr', float('-inf')) < float(nms_thr):
                            cfg_for_init.model.test_cfg.rpn['nms_thr'] = float(nms_thr)
                    if nms_pre is not None:
                        cfg_for_init.model.test_cfg.rpn['nms_pre'] = int(nms_pre)
                    if max_num is not None:
                        cfg_for_init.model.test_cfg.rpn['nms_post'] = int(max_num)
                else: # VoxelNet and others
                    if nms_thr is not None:
                        cfg_for_init.model.test_cfg['nms_thr'] = float(nms_thr)
                    if nms_pre is not None:
                        cfg_for_init.model.test_cfg['nms_pre'] = int(nms_pre)
                    if max_num is not None:
                        cfg_for_init.model.test_cfg['max_num'] = int(max_num)
            except Exception:
                # If the cfg structure is unexpected, skip overriding (keep native cfg).
                pass
        
        self.detector3d = init_model(cfg_for_init, checkpoint, device=self.device)
        
        cfg = self.detector3d.cfg.copy()

        # Box type/mode expected by mmdet3d pipelines and model.
        self.box_type_3d, self.box_mode_3d = get_box_type(
            cfg.test_dataloader.dataset.box_type_3d
        )

        # Default pipeline (expects to load points from file).
        test_pipeline_3d = deepcopy(cfg.test_dataloader.dataset.pipeline)
        self.test_pipeline_3d = ComposeMMEngine(test_pipeline_3d)

        # Pipeline variant for pre-loaded points.
        # Mirrors the official mmdet3d inference behavior:
        # - replace the first points loader with LoadPointsFromDict
        # - remove any sweeps loading step (we may load sweeps manually)
        test_pipeline_3d_dict = deepcopy(cfg.test_dataloader.dataset.pipeline)
        # Replace the first LoadPoints* step (usually index 0).
        first_points_loader_idx = None
        for i, t in enumerate(test_pipeline_3d_dict):
            t_type = t.get('type') if isinstance(t, dict) else None
            if isinstance(t_type, str) and t_type.startswith('LoadPointsFrom'):
                first_points_loader_idx = i
                break
        if first_points_loader_idx is not None:
            test_pipeline_3d_dict[first_points_loader_idx]['type'] = 'LoadPointsFromDict'
        else:
            # Fallback: assume index 0 is the points loader.
            if len(test_pipeline_3d_dict) > 0 and isinstance(test_pipeline_3d_dict[0], dict):
                test_pipeline_3d_dict[0]['type'] = 'LoadPointsFromDict'

        def _is_sweeps_loader(step: dict) -> bool:
            if not isinstance(step, dict):
                return False
            t = step.get('type')
            return isinstance(t, str) and ('MultiSweeps' in t or 'Sweeps' in t)

        test_pipeline_3d_dict = [s for s in test_pipeline_3d_dict if not _is_sweeps_loader(s)]
        self.test_pipeline_3d_from_dict = ComposeMMEngine(test_pipeline_3d_dict)
    
    def _init_detector2d(self):
        """Initialize the 2D detector."""
        if self.model_type == 'ultralytics':
            model_path = self.detector2d_cfg.get('model_path', self.detector2d_cfg.get('checkpoint_path', 'yolov8n.pt'))
            self.detector2d = YOLO(model_path)
            self.test_pipeline_2d = None
        else:
            cfg_file = self.detector2d_cfg.get('cfg_path', None)
            checkpoint = self.detector2d_cfg.get('checkpoint_path', None)

            # Optional: override detector test-time NMS parameters from late-fusion cfg.
            nms_thr = self.detector2d_cfg.get('nms_thr', None)
            nms_pre = self.detector2d_cfg.get('nms_pre', None)
            max_num = self.detector2d_cfg.get('max_num', None)
            nms_type = self.detector2d_cfg.get('nms_type', None)

            cfg_for_init = cfg_file
            if cfg_file is not None and any(v is not None for v in [nms_thr, nms_pre, max_num, nms_type]):
                cfg_for_init = Config.fromfile(cfg_file)
                try:
                    # Two-stage (e.g., Faster R-CNN): cfg.model.test_cfg.rcnn.*
                    if hasattr(cfg_for_init.model.test_cfg, 'rcnn'):
                        if nms_thr is not None and hasattr(cfg_for_init.model.test_cfg.rcnn, 'nms'):
                            cfg_for_init.model.test_cfg.rcnn.nms.iou_threshold = float(nms_thr)
                        if nms_type is not None and hasattr(cfg_for_init.model.test_cfg.rcnn, 'nms'):
                            cfg_for_init.model.test_cfg.rcnn.nms.type = str(nms_type)
                        if max_num is not None:
                            # mmdet uses max_per_img for final per-image cap.
                            cfg_for_init.model.test_cfg.rcnn.max_per_img = int(max_num)
                        if nms_pre is not None and hasattr(cfg_for_init.model.test_cfg, 'rpn'):
                            cfg_for_init.model.test_cfg.rpn.nms_pre = int(nms_pre)
                    else:
                        # One-stage (e.g., RetinaNet): cfg.model.test_cfg.*
                        if nms_thr is not None and hasattr(cfg_for_init.model.test_cfg, 'nms'):
                            cfg_for_init.model.test_cfg.nms.iou_threshold = float(nms_thr)
                        if nms_type is not None and hasattr(cfg_for_init.model.test_cfg, 'nms'):
                            cfg_for_init.model.test_cfg.nms.type = str(nms_type)
                        if max_num is not None and hasattr(cfg_for_init.model.test_cfg, 'max_per_img'):
                            cfg_for_init.model.test_cfg.max_per_img = int(max_num)
                except Exception:
                    # If the cfg structure is unexpected, skip overriding (keep native cfg).
                    pass

            self.detector2d = init_detector(cfg_for_init, checkpoint, device=self.device)
            
            test_pipeline_cfg = get_test_pipeline_cfg(self.detector2d.cfg)
            # test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
            self.test_pipeline_2d = ComposeMMCV(test_pipeline_cfg)
    
    def _init_frustum_detector(self):
        """Initialize the frustum detector for detection recovery."""
        cfg_file = self.frustum_detector_cfg.get('cfg_path', None)
        checkpoint = self.frustum_detector_cfg.get('checkpoint_path', None)

        self.frustum_detector = init_model(cfg_file, checkpoint, device=self.device)

        cfg = self.frustum_detector.cfg.copy()
        test_pipeline_3d_frustum = deepcopy(cfg.test_dataloader.dataset.pipeline)
        self.test_pipeline_3d_frustum = ComposeMMEngine(test_pipeline_3d_frustum)
    
    def rgb_branch_inference(self, img_files: Union[str, List[str]]) -> List[Tuple[Tensor, Tensor, Tensor, Tuple, Tensor]]:
        """
        Run 2D detection on one or more images in batch.
        
        Args:
            img_files: Single image file path or list of image file paths.
        
        Returns:
            List of tuples (bboxes_2d, labels_2d, scores_2d, img_shape, masks_2d) for each image.
            masks_2d is a tensor of shape (N, H, W) or None if instance segmentation is not available.
        """
        # Convert single image to list
        if isinstance(img_files, str):
            img_files = [img_files]
        
        if self.model_type == 'ultralytics':
            # Batch inference for ultralytics
            batch_detection_results = self.detector2d.predict(
                source=img_files, 
                save=False, 
                verbose=False,
                stream=False,
                device=self.device,
                conf=self.score_thr_2d.min().item())
            
            results = []
            for detection_results in batch_detection_results:
                bboxes = detection_results.boxes.xyxy
                labels = detection_results.boxes.cls.long()
                scores = detection_results.boxes.conf
                img_shape = detection_results.orig_shape
                
                # Extract masks if available (instance segmentation)
                masks = None
                if hasattr(detection_results, 'masks') and detection_results.masks is not None:
                    masks = detection_results.masks.data
                
                # Filter by valid classes
                valid_boxes = torch.isin(labels, self.valid_2d_classes)
                scores = scores[valid_boxes]
                bboxes = bboxes[valid_boxes]
                labels = labels[valid_boxes]
                if masks is not None:
                    masks = masks[valid_boxes]
                labels = self.img_label_mapping[labels].long()

                # Filter by score threshold
                filter_scores = scores >= self.score_thr_2d[labels]
                bboxes = bboxes[filter_scores]
                labels = labels[filter_scores]
                scores = scores[filter_scores]
                if masks is not None:
                    masks = masks[filter_scores]
                
                results.append((bboxes, labels, scores, img_shape, masks))
        
        else:
            # Batch inference for mmdet
            batch_inputs = []
            for img_id, img_file in enumerate(img_files):
                inputs = dict(img_path=img_file, img_id=img_id)
                inputs = self.test_pipeline_2d(inputs)
                batch_inputs.append(inputs)
            
            # Prepare batch data
            batch_data = {
                'inputs': [inp['inputs'] for inp in batch_inputs],
                'data_samples': [inp['data_samples'] for inp in batch_inputs]
            }
            
            with torch.no_grad():
                batch_detection_results = self.detector2d.test_step(batch_data)
            
            results = []
            for i, detection_results in enumerate(batch_detection_results):
                labels = detection_results.pred_instances.labels
                scores = detection_results.pred_instances.scores
                bboxes = detection_results.pred_instances.bboxes
                img_shape = batch_data['data_samples'][i].ori_shape
                
                # Extract masks if available (instance segmentation)
                masks = None
                if hasattr(detection_results.pred_instances, 'masks') and detection_results.pred_instances.masks is not None:
                    masks = detection_results.pred_instances.masks

                # Filter by valid classes
                valid_boxes = torch.isin(labels, self.valid_2d_classes)
                scores = scores[valid_boxes]
                bboxes = bboxes[valid_boxes]
                labels = labels[valid_boxes]
                if masks is not None:
                    masks = masks[valid_boxes]
                labels = self.img_label_mapping[labels].long()

                # Filter by score threshold
                filter_scores = scores >= self.score_thr_2d[labels]
                bboxes = bboxes[filter_scores]
                labels = labels[filter_scores]
                scores = scores[filter_scores]
                if masks is not None:
                    masks = masks[filter_scores]
                
                results.append((bboxes, labels, scores, img_shape, masks))
        
        return results
    
    def lidar_branch_inference(self, pc_file: str, points: np.ndarray = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray, Dict]:
        """
        Run 3D detection on the point cloud.
        
        Args:
            pc_file: Path to point cloud file.
            points: Optional pre-loaded point cloud array. If None, will load from pc_file.
        
        Returns:
            Tuple of (bboxes_3d, corners_3d, labels_3d, scores_3d, points, collate_data)
        """
        points_is_empty = False
        if points is not None:
            try:
                points_is_empty = (points.size == 0)
            except Exception:
                points_is_empty = False

        if points is not None and not points_is_empty:
            # Pre-loaded points: use a pipeline that starts with LoadPointsFromDict.
            pipeline = getattr(self, 'test_pipeline_3d_from_dict', None) or self.test_pipeline_3d
            inputs = dict(
                points=points,
                timestamp=1,
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            # No points provided: use the original pipeline and let it load from file.
            pipeline = self.test_pipeline_3d
            inputs = dict(
                lidar_points=dict(lidar_path=pc_file),
                timestamp=1,
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)

        collate_data = pseudo_collate([pipeline(inputs)])

        # Always return the *post-pipeline* points (they may be filtered/augmented
        # by the pipeline, e.g., FOV filtering or feature selection).
        points_ret = collate_data['inputs']['points'][0]
        
        with torch.no_grad():
            detection_output = self.detector3d.test_step(collate_data)[0]

        bboxes = detection_output.pred_instances_3d.bboxes_3d.tensor
        corners = detection_output.pred_instances_3d.bboxes_3d.corners
        labels = detection_output.pred_instances_3d.labels_3d.long()
        scores = detection_output.pred_instances_3d.scores_3d

        # # Filter detector outputs to only keep allowed detector labels.
        # if getattr(self, 'valid_labels_3d', None) is not None:
        #     valid_mask = torch.isin(labels, self.valid_labels_3d)
        #     bboxes = bboxes[valid_mask]
        #     corners = corners[valid_mask]
        #     labels = labels[valid_mask]
        #     scores = scores[valid_mask]

        # # Remap detector labels to evaluator ordering.
        # if getattr(self, 'class_mapping_3d', None) is not None and labels.numel() > 0:
        #     max_label = int(labels.max().item())
        #     if max_label >= int(self.class_mapping_3d.numel()):
        #         raise ValueError(
        #             f"class_mapping_3d is too short (len={int(self.class_mapping_3d.numel())}) "
        #             f"for detector label id {max_label}."
        #         )
        #     labels = self.class_mapping_3d[labels].long()

        # if labels.numel() > 0:
        #     max_mapped = int(labels.max().item())
        #     if max_mapped >= int(self.score_thr_3d.numel()):
        #         raise ValueError(
        #             f"score_thr_3d is too short (len={int(self.score_thr_3d.numel())}) "
        #             f"for mapped label id {max_mapped}."
        #         )
        
        filter_scores = scores >= self.score_thr_3d[labels]
        return bboxes[filter_scores], corners[filter_scores], labels[filter_scores], scores[filter_scores], points_ret, collate_data
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Abstract method for prediction. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def visualize_predict(self, *args, **kwargs):
        """
        Abstract method for prediction with visualization. Must be implemented by subclasses.
        """
        pass
