import torch
from torch import Tensor
from typing_extensions import List, Dict, Tuple
from mmengine.model import BaseModule
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.det_data_sample import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.models.roi_heads import Base3DRoIHead
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.structures.bbox_3d import (
    rotation_3d_in_axis, 
    xywhr2xyxyr, 
    BaseInstance3DBoxes, 
    LiDARInstance3DBoxes, 
    DepthInstance3DBoxes,
    Box3DMode
)


@MODELS.register_module()
class LateFusionBEVRoIHead(Base3DRoIHead):
    
    def __init__(self,
                 img_roi_layer: dict = None,
                 lidar_bev_roi_layer: dict = None,
                 bbox_head: dict = None,
                 iou_calculator_2d: dict = dict(type='mmdet.BboxOverlaps2D'),
                 match_iou_threshold: int = 0.5,
                 assign_per_class: bool = False,
                 match_per_class: bool = True,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None):
        self.lidar_bev_roi_layer = lidar_bev_roi_layer
        self.img_roi_layer = img_roi_layer
        self.bbox_head = bbox_head
        self.sampling = bbox_head['loss_cls']['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        super(LateFusionBEVRoIHead, self).__init__(
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # self.img_roi_layer = MODELS.build(img_roi_layer)
        # self.lidar_bev_roi_layer = MODELS.build(lidar_bev_roi_layer)
        # self.bbox_head = MODELS.build(bbox_head)
        self.iou_calculator_2d = TASK_UTILS.build(iou_calculator_2d)   
        self.match_iou_threshold = match_iou_threshold
        self.assign_per_class = assign_per_class
        self.match_per_class = match_per_class
        
    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.img_roi_layer = MODELS.build(self.img_roi_layer)
        self.lidar_bev_roi_layer = MODELS.build(self.lidar_bev_roi_layer)
        self.bbox_head = MODELS.build(self.bbox_head)
        
    def init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        # if self.sampling:
        #     self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        # else:
        #     # do not sample
        #     self.bbox_sampler = PseudoSampler()
        self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]
        
    def forward(self, 
                img_features: List[torch.Tensor], 
                lidar_features: List[torch.Tensor],
                batch_data_samples: List[Det3DDataSample]) -> Tuple:
        
        fused_features, lidar_rois, img_rois, proposals_per_sample = self._extract_roi_matching_features(
            img_features,
            lidar_features,
            batch_data_samples)
        cls_scores, bbox_preds = self.bbox_head(fused_features)
        # todo: postprocessing?
        return cls_scores, bbox_preds
    
    def predict(self, 
                img_features: List[torch.Tensor],
                lidar_features: List[torch.Tensor],
                batch_data_samples: List[Det3DDataSample]) -> List[Det3DDataSample]:
        
        batch_input_metas = [data_samples.metainfo for data_samples in batch_data_samples]
                
        fused_features, lidar_rois, img_rois, proposals_per_sample = self._extract_roi_matching_features(
            img_features,
            lidar_features,
            batch_data_samples)
        cls_scores, bbox_preds = self.bbox_head(fused_features)
        cls_scores = cls_scores.split(proposals_per_sample, 0)
        if bbox_preds is not None:
            bbox_preds = bbox_preds.split(proposals_per_sample, 0)
        lidar_rois = lidar_rois.split(proposals_per_sample, 0)
        result_list = self.bbox_head.predict_by_feat(
            rois=lidar_rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_input_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=False)
        return result_list
    
    def loss(self, 
             img_features: List[torch.Tensor], 
             lidar_features: List[torch.Tensor],
             batch_data_samples: List[Det3DDataSample]) -> List[Det3DDataSample]:
        
        fused_features, lidar_rois, img_rois, proposals_per_sample = self._extract_roi_matching_features(
            img_features,
            lidar_features,
            batch_data_samples)
        cls_scores, bbox_preds = self.bbox_head(fused_features)
        
        batch_gt_instances = [data_sample.gt_instances_3d for data_sample in batch_data_samples]
        batch_input_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        
        lidar_rois_split = lidar_rois.split(proposals_per_sample, 0)
        
        num_samples = len(batch_data_samples)
        sampling_results = []
        for i in range(num_samples):
            # todo: assign by class (if parameter is True)         
            rpn_results_3d = InstanceData()
            rpn_results_3d.priors = lidar_rois_split[i][:, 1:]
            
            gt_instances = InstanceData()
            gt_instances.bboxes_3d = batch_gt_instances[i].bboxes_3d.tensor
            gt_instances.labels_3d = batch_gt_instances[i].labels_3d

            assign_result = self.bbox_assigner.assign(
                rpn_results_3d, gt_instances)
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results_3d,
                gt_instances)
            sampling_results.append(sampling_result)
            
        # this is valid since sampler is a PseudoSampler, need to change otherwise?
        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=cls_scores,
            bbox_pred=bbox_preds,
            rois=lidar_rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        
        losses = dict()
        losses.update(bbox_loss_and_target['loss_bbox'])
        return losses
            
        
    def _extract_roi_matching_features(self,
                                       img_features: List[torch.Tensor], 
                                       lidar_features: List[torch.Tensor],
                                       batch_data_samples: List[Det3DDataSample]):
        
        img_proposals = []
        img_proposals_scaled = []
        for data_samples in batch_data_samples:
            bboxes = data_samples.pred_instances.bboxes
            img_proposals.append(bboxes)
            if bboxes.shape[0] > 0:
                scale_factor = bboxes.new_tensor(
                    data_samples.metainfo['scale_factor']).repeat((1, 2))
                bboxes_scaled = (
                    bboxes.view(bboxes.size(0), -1, 4) *
                    scale_factor).view(bboxes.size()[0], -1)
                img_proposals_scaled.append(bboxes_scaled)
            else:
                img_proposals_scaled.append(bboxes)
        img_rois = bbox2roi(img_proposals_scaled)
        img_pooled_features = self.img_roi_layer(img_features, img_rois)
        
        lidar_proposals = [data_samples.pred_instances_3d.bboxes_3d for data_samples in batch_data_samples]
        lidar_proposals_bev = [proposal.bev for proposal in lidar_proposals]
        lidar_rois_bev = bbox2roi(lidar_proposals_bev)
        lidar_pooled_features = self.lidar_bev_roi_layer(lidar_features, lidar_rois_bev)
        
        fused_features, lidar_proposals, img_proposals, batch_ids = self._fuse_features(
            lidar_pooled_features, 
            lidar_proposals,
            img_pooled_features, 
            img_proposals,
            batch_data_samples)
            
        return fused_features, lidar_proposals, img_proposals, batch_ids
    
    def _fuse_features(self,
                       lidar_roi_features: Tensor,
                       lidar_proposals: List[Tensor],
                       img_roi_features: Tensor,
                       img_proposals: List[Tensor],
                       batch_data_samples: List[Det3DDataSample]):
        
        features_matches = []
        matches_proposals_lidar = []
        matches_proposals_img = []
        matches_per_sample = []
        start_batch_id_lidar = 0
        start_batch_id_img = 0
        features_channels = img_roi_features.shape[1] + lidar_roi_features.shape[1]
        for i in range(len(lidar_proposals)):
            
            projected_bboxes = self._project_3d_to_2d_single(lidar_proposals[i], batch_data_samples[i].metainfo)
            
            end_batch_id_lidar = start_batch_id_lidar + lidar_proposals[i].shape[0]
            end_batch_id_img = start_batch_id_img + img_proposals[i].shape[0]
            
            overlaps = self.iou_calculator_2d(projected_bboxes, img_proposals[i])
            
            # semantic consistency: only boxes with same class can match
            if self.match_per_class:
                lidar_labels = batch_data_samples[i].pred_instances_3d.labels_3d
                img_labels = batch_data_samples[i].pred_instances.labels
                class_mask = (lidar_labels.unsqueeze(1) == img_labels.unsqueeze(0)).float()
                overlaps = overlaps * class_mask
            
            lidar_ids, img_ids = torch.nonzero(overlaps > self.match_iou_threshold, as_tuple=True)        
            
            if lidar_ids.shape[0] > 0:
                features = torch.cat((lidar_roi_features[start_batch_id_lidar:end_batch_id_lidar][lidar_ids], 
                                      img_roi_features[start_batch_id_img:end_batch_id_img][img_ids]), dim=1)
                matches_proposals_lidar.append(lidar_proposals[i][lidar_ids].tensor)
                matches_proposals_img.append(img_proposals[i][img_ids])
                matches_per_sample.append(lidar_ids.shape[0])
                features_matches.append(features)
            else:
                matches_proposals_lidar.append(torch.empty((0, 7), dtype=torch.float32, device=lidar_proposals[i].device))
                matches_proposals_img.append(torch.empty((0, 4), dtype=torch.float32, device=img_proposals[i].device))
                matches_per_sample.append(0)
                features_matches.append(torch.empty((0, features_channels, *lidar_roi_features.shape[2:]), dtype=torch.float32, device=lidar_roi_features.device))
                
            start_batch_id_lidar = end_batch_id_lidar
            start_batch_id_img = end_batch_id_img
                            
        return (
            torch.vstack(features_matches),
            bbox2roi(matches_proposals_lidar),
            bbox2roi(matches_proposals_img),
            matches_per_sample
        )
            
    
    def _project_3d_to_2d_single(self, 
                                 bboxes_3d: BaseInstance3DBoxes, 
                                 img_metas: Det3DDataSample):
        """Project 3D bounding boxes to 2D image plane."""
        
        # TODO: support also the box 3d depth mode?  
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            bboxes_3d_cam = bboxes_3d.convert_to(Box3DMode.CAM, rt_mat=img_metas['lidar2cam'])
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d_cam = bboxes_3d.convert_to(Box3DMode.CAM, rt_mat=img_metas['depth2cam'])
        else:
            bboxes_3d_cam = bboxes_3d

        corners = bboxes_3d_cam.corners
        corners = torch.cat([corners, torch.ones_like(corners[:, :, :1])], dim=-1)

        num_boxes = corners.shape[0]
        P = corners.new_tensor(img_metas['cam2img'][:3, :])
        projection_matrix = P.unsqueeze(0).expand(num_boxes, -1, -1)
        
        corners_projected = torch.matmul(corners, projection_matrix.transpose(1, 2))
        corners_projected[:, :, :2] /= corners_projected[:, :, 2:]
        
        min_x = torch.min(corners_projected[:, :, 0], dim=1)[0]
        min_y = torch.min(corners_projected[:, :, 1], dim=1)[0]
        max_x = torch.max(corners_projected[:, :, 0], dim=1)[0]
        max_y = torch.max(corners_projected[:, :, 1], dim=1)[0]
        
        return torch.stack((min_x, min_y, max_x, max_y), dim=1)
    
    def _convert_to_xyxy(self, rois: Tensor):
        '''From (x,y,w,h,yaw) to (x1,y1,x2,y2)'''
        half_dimensions = rois[:, 2:4].reshape(-1, 1, 2).repeat(1, 4, 1) / 2
        centers = rois[:, :2].reshape(-1, 1, 2).repeat(1, 4, 1)
        
        # directions for the yaw rotation
        direction_1 = torch.cos(rois[:, -1])
        direction_2 = torch.sin(rois[:, -1])
    
        corners_relative = self.corners_multipliers * half_dimensions
        rotation_matrices = torch.zeros((rois.shape[0], 2, 2), dtype=torch.float32, device=rois.device)
        rotation_matrices[:, 0, 0] = direction_1.view(-1)
        rotation_matrices[:, 0, 1] = -direction_2.view(-1)
        rotation_matrices[:, 1, 0] = direction_2.view(-1)
        rotation_matrices[:, 1, 1] = direction_1.view(-1)

        rotated_corners = torch.bmm(rotation_matrices, corners_relative.permute(0, 2, 1)).permute(0, 2, 1)
        rotated_corners = torch.squeeze(rotated_corners)
        corners = centers + rotated_corners
        
        # torch.min returns (min, argmin), argmin is not useful here
        min_x, _ = torch.min(corners[:, :, 0], dim=1)
        min_y, _ = torch.min(corners[:, :, 1], dim=1)
        max_x, _ = torch.max(corners[:, :, 0], dim=1)
        max_y, _ = torch.max(corners[:, :, 1], dim=1)
        
        return torch.stack((min_x, min_y, max_x, max_y), dim=1)