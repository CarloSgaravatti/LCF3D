import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmcv.ops import box_iou_rotated

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)
from mmdet3d.structures import get_box_type
from ..functional.kitti_utils.rotate_iou import rotate_iou_gpu_eval, rotate_iou_kernel_eval


@METRICS.register_module()
class FrustumLocalizationMetric(BaseMetric):
    def __init__(self,
                 box_type_3d: str = 'LiDAR',
                 classes: list = ['Pedestrian', 'Cyclist', 'Car'],
                 with_cls_out: bool = False,
                 with_background_score: bool = False,
                 iou_thresholds = [0.5, 0.5, 0.7],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(FrustumLocalizationMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.box_type_3d = box_type_3d
        self.classes = classes
        self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32, device='cpu')
        self.with_cls_out = with_cls_out
        self.with_background_score = with_background_score
                
    def process(self, data_batch: dict, outputs: list) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        if self.with_cls_out:
            data_samples, labels, bboxes, scores = outputs
        elif self.with_background_score:
            data_samples, bboxes, scores = outputs
        else:
            data_samples, bboxes = outputs
        labels_gt = []
        bboxes_gt = []
        for data_sample in data_samples:
            pred_3d = data_sample.eval_ann_info
            bboxes_gt.append(pred_3d['gt_bboxes_3d'].tensor[0:1].to('cpu'))
            labels_gt.append(pred_3d['gt_labels_3d'][0])
        bboxes_gt = torch.cat(bboxes_gt, dim=0).to('cpu')
        labels_gt = torch.tensor(labels_gt, dtype=torch.int32, device='cpu')
        if self.with_cls_out:
            self.results.append((bboxes.to('cpu'), labels.to('cpu'), scores.to('cpu'), bboxes_gt, labels_gt))
        elif self.with_background_score:
            self.results.append((bboxes.to('cpu'), scores.to('cpu'), bboxes_gt, labels_gt))
        else:
            self.results.append((bboxes.to('cpu'), bboxes_gt, labels_gt))
        
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        bboxes_pred = torch.cat([r[0] for r in results], dim=0)
        if self.with_cls_out:
            labels_pred = torch.cat([r[1] for r in results], dim=0)
            scores_pred = torch.cat([r[2] for r in results], dim=0)
        elif self.with_background_score:
            scores_pred = torch.cat([r[1] for r in results], dim=0)
        bboxes_gt = torch.cat([r[-2] for r in results], dim=0)
        labels_gt = torch.cat([r[-1] for r in results], dim=0)
        
        if self.with_background_score:
            background_samples = labels_gt == len(self.classes)
            lb = (~background_samples).int()
            background_accuracy = torch.sum((scores_pred > 0.5).int() == lb) / scores_pred.shape[0]
            if self.with_cls_out:
                labels_pred = labels_pred[~background_samples]
            if self.with_cls_out or self.with_background_score:
                scores_pred = scores_pred[~background_samples]
            bboxes_pred = bboxes_pred[~background_samples]
            labels_gt = labels_gt[~background_samples]
            bboxes_gt = bboxes_gt[~background_samples]
        box_type_3d, box_mode_3d = get_box_type(self.box_type_3d)
        
        # only the fact that are in camera coords is important, the 
        # extrinsic reference system is not important
        bboxes_gt_ = box_type_3d(bboxes_gt)
        bboxes_pred_ = box_type_3d(bboxes_pred)
        
        centers_gt = bboxes_gt_.center
        centers_pred = bboxes_pred_.center
        yaw_gt = bboxes_gt_.yaw
        yaw_pred = bboxes_pred_.yaw
        
        center_distances = torch.sqrt(torch.sum((centers_gt - centers_pred)**2, dim=-1))
        angle_diff = torch.abs(((yaw_gt - yaw_pred + torch.pi) % (2 * torch.pi)) - torch.pi)
        sin_diff = torch.sin(yaw_gt - yaw_pred)
        
        ious_bev = box_iou_rotated(bboxes_gt_.bev, bboxes_pred_.bev, aligned=True)
        
        top_heights = torch.cat([
            bboxes_pred_.top_height.unsqueeze(1),
            bboxes_gt_.top_height.unsqueeze(1),
        ], dim=1)
        bottom_heights = torch.cat([
            bboxes_pred_.bottom_height.unsqueeze(1),
            bboxes_gt_.bottom_height.unsqueeze(1),
        ], dim=1)
        
        intersection = torch.min(top_heights, dim=1)[0] - torch.max(bottom_heights, dim=1)[0]
        union = torch.max(top_heights, dim=1)[0] - torch.min(bottom_heights, dim=1)[0]
        height_iou = intersection / union
        
        ious_3d = ious_bev * height_iou
        
        thrs = self.iou_thresholds[labels_gt]
        iou_mask = ious_3d > thrs
        if self.with_cls_out:
            cls_mask = labels_gt == labels_pred
        
        ret_dict = dict(
            IoU_Accuracy=torch.sum(iou_mask) / labels_gt.shape[0],
            IoU_BEV=torch.mean(ious_bev),
            IoU_3D=torch.mean(ious_3d),
            Center_Distance=torch.mean(center_distances),
            Heading_Sin_Diff=torch.mean(sin_diff),
            Heading_Angle_Diff=torch.mean(angle_diff),
        )
        if self.with_cls_out:
            ret_dict['Accuracy'] = torch.sum(cls_mask) / labels_pred.shape[0]
            ret_dict['IoU_Class_Accuracy'] = torch.sum(cls_mask * iou_mask) / labels_pred.shape[0]
        elif self.with_background_score:
            ret_dict['BackgroundAccuracy'] = background_accuracy
            
        for i, cls_name in enumerate(self.classes):
            filter_cls_gt = labels_gt == i
            ret_dict[f'{cls_name}_IoU_BEV'] = torch.mean(ious_bev[filter_cls_gt])
            ret_dict[f'{cls_name}_IoU_3D'] = torch.mean(ious_3d[filter_cls_gt])
            ret_dict[f'{cls_name}_Center_Distance_AVG'] = torch.mean(center_distances[filter_cls_gt])
            
            if self.with_cls_out:
                filter_cls_pred = labels_pred == i
                ret_dict[f'{cls_name}_Recall'] = torch.sum(labels_gt[filter_cls_gt] == labels_pred[filter_cls_gt]) / torch.sum(filter_cls_gt)
                ret_dict[f'{cls_name}_Precision'] = torch.sum(labels_gt[filter_cls_pred] == labels_pred[filter_cls_pred]) / torch.sum(filter_cls_pred)
            if self.with_background_score:
                ret_dict[f'{cls_name}_AVG_Score'] = torch.mean(scores_pred[filter_cls_gt])
        
        return ret_dict
