import warnings
import numpy as np
from typing_extensions import List, Tuple, Optional, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType, InstanceList
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.structures.bbox import get_box_tensor
from mmdet.models.utils import empty_instances, multi_apply
from mmengine.model import BaseModule
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.registry import MODELS, TASK_UTILS


@MODELS.register_module()
class Residual3DBboxHead(ConvFCBBoxHead):
    
    def __init__(self,
                 num_classes: int = 3,
                 in_channels: int = 256,
                 with_avg_pool: bool = False,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 reg_class_agnostic: bool = True,
                 roi_feat_size: int = 7,
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 predict_box_type: str = 'hbox',
                 reg_decoded_bbox: bool = False,
                 reg_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 cls_predictor_cfg: ConfigType = dict(type='mmdet.Linear'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 num_shared_convs: int = 0,
                 num_shared_fcs: int = 0,
                 num_cls_convs: int = 0,
                 num_cls_fcs: int = 0,
                 num_reg_convs: int = 0,
                 num_reg_fcs: int = 0,
                 conv_out_channels: int = 256,
                 fc_out_channels: int = 1024,
                 conv_cfg: Optional[Union[dict, ConfigDict]] = None,
                 norm_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: OptMultiConfig = None,
                 dir_limit_offset: int = 0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None):
        super(Residual3DBboxHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            with_avg_pool=with_avg_pool,
            with_cls=with_cls,
            with_reg=with_reg,
            roi_feat_size=roi_feat_size,
            bbox_coder=bbox_coder,
            predict_box_type=predict_box_type,
            reg_class_agnostic=reg_class_agnostic,
            reg_decoded_bbox=reg_decoded_bbox,
            reg_predictor_cfg=reg_predictor_cfg,
            cls_predictor_cfg=cls_predictor_cfg,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            num_shared_convs=num_shared_convs,
            num_shared_fcs=num_shared_fcs,
            num_cls_convs=num_cls_convs,
            num_cls_fcs=num_cls_fcs,
            num_reg_convs=num_reg_convs,
            num_reg_fcs=num_reg_fcs,
            conv_out_channels=conv_out_channels,
            fc_out_channels=fc_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.box_code_size = self.bbox_coder.code_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.dir_limit_offset = dir_limit_offset
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')       

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
            
    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        bbox_preds: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg: Optional[ConfigDict] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            rois (tuple[Tensor]): Tuple of boxes to be transformed.
                Each has shape  (num_boxes, 5). last dimension 5 arrange as
                (batch_index, x1, y1, x2, y2).
            cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
                (num_boxes, num_classes + 1).
            bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
                has shape (num_boxes, num_classes * 4).
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        
        if self.with_reg and self.with_cls:
            assert len(cls_scores) == len(bbox_preds)
            
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id] if self.with_cls else None,
                bbox_pred=bbox_preds[img_id] if self.with_reg else None,
                img_meta=img_meta,
                rescale=rescale,
                rcnn_test_cfg=rcnn_test_cfg)
            result_list.append(results)

        return result_list
    
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Has a shape (num_instances, 7),
                  the last dimension 4 arrange as (x, y, z, w, h, l, r).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            results.bboxes_3d = img_meta['box_type_3d'](torch.zeros((0, self.bbox_coder.encode_size), device=roi.device), 
                                                        box_dim=self.box_code_size)
            if rcnn_test_cfg is None:
                results.scores_3d = torch.zeros((0, self.num_classes + 1), device=roi.device)
            else:
                results.scores_3d = torch.zeros((0,), device=roi.device)
            results.labels_3d = torch.zeros((0, ),
                                         device=roi.device,
                                         dtype=torch.long)
            return results

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(roi[..., 1:], bbox_pred)
        else:
            bboxes = roi[:, 1:].clone()
            
        bboxes_for_nms = xywhr2xyxyr(img_meta['box_type_3d'](
            bboxes, box_dim=self.box_code_size).bev)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        bboxes = bboxes.view(num_rois, -1)
        
        bboxes_3d, scores_3d, labels_3d = box3d_multiclass_nms(
            bboxes,
            bboxes_for_nms,
            scores,
            rcnn_test_cfg.score_thr,
            rcnn_test_cfg.max_num,
            rcnn_test_cfg,
        )
        bboxes_3d = img_meta['box_type_3d'](bboxes_3d, box_dim=self.box_code_size)
        # TODO: check if it is needed to postprocess the yaw angle
        results.bboxes_3d = bboxes_3d
        results.scores_3d = scores_3d
        results.labels_3d = labels_3d
        return results
    
    def get_targets(self,
                    sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict,
                    concat: bool = True) -> tuple:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_targets_single` function.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

            - labels (list[Tensor],Tensor): Gt_labels for all
                proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - label_weights (list[Tensor]): Labels_weights for
                all proposals in a batch, each tensor in list has
                shape (num_proposals,) when `concat=False`, otherwise
                just a single tensor has shape (num_all_proposals,).
            - bbox_targets (list[Tensor],Tensor): Regression target
                for all proposals in a batch, each tensor in list
                has shape (num_proposals, 4) when `concat=False`,
                otherwise just a single tensor has shape
                (num_all_proposals, 4), the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            - bbox_weights (list[tensor],Tensor): Regression weights for
                all proposals in a batch, each tensor in list has shape
                (num_proposals, 4) when `concat=False`, otherwise just a
                single tensor has shape (num_all_proposals, 4).
        """
        pos_priors_list = [res.pos_priors for res in sampling_results]
        neg_priors_list = [res.neg_priors for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_targets_single,
            pos_priors_list,
            neg_priors_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_inds_list,
            neg_inds_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
    
    def _get_targets_single(self, pos_priors: Tensor, neg_priors: Tensor,
                            pos_gt_bboxes: Tensor, pos_gt_labels: Tensor,
                            pos_inds: Tensor, neg_inds: Tensor,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_priors.size(0)
        num_neg = neg_priors.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_priors.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        reg_dim = pos_gt_bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        label_weights = pos_priors.new_zeros(num_samples)
        bbox_targets = pos_priors.new_zeros(num_samples, reg_dim)
        bbox_weights = pos_priors.new_zeros(num_samples, reg_dim)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_priors, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights