from typing_extensions import Sequence, Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_conv_layer
from mmengine.model import BaseModule

from mmdet.models.utils import empty_instances, multi_apply
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.structures.bbox import get_box_tensor

from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from mmdet3d.structures.det3d_data_sample import SampleList, Det3DDataSample
from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.models.dense_heads.train_mixins import get_direction_target


@MODELS.register_module()
class FrustumConvNetBboxHead(BaseModule):
    
    def __init__(self,
                 point_cloud_range: Sequence[int],
                 num_classes: int = 3,
                 reg_decoded_bbox: bool = False,
                 anchor_generator: ConfigType = dict(
                     type='FrustumAnchorGenerator',
                     frustum_range=[0, -40, -3, 70.4, 40, 1],
                     height=-0.6,
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57]),
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 reg_conv_channels=(),
                 dir_conv_channels=(),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 with_dir_cls: bool = True,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: int = 0,
                 init_cfg=None,
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 *args,
                 **kwargs):
        super(FrustumConvNetBboxHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        self.num_anchors = self.prior_generator.num_base_anchors
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size
        self.point_cloud_range = point_cloud_range
        self.reg_decoded_bbox = reg_decoded_bbox
        
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.num_classes = num_classes + 1 if not self.use_sigmoid_cls else num_classes
        self.num_reg_out_channels = self.bbox_coder.code_size * self.num_anchors
        self.num_cls_out_channels = self.num_classes * self.num_anchors
        self.num_dir_out_channels = 2 * self.num_anchors
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_dir_cls = with_dir_cls
        self.diff_rad_by_sin = diff_rad_by_sin
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_cls = MODELS.build(loss_cls)
        if with_dir_cls:
            self.loss_dir = MODELS.build(loss_dir)
        self._init_assigner_sampler()
        
        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.reg_conv_channels = reg_conv_channels
        self.dir_conv_channels = dir_conv_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.bias = bias

        # add shared convs
        if len(self.shared_conv_channels) > 0:
            self.shared_convs = self._add_conv_branch(
                self.in_channels, self.shared_conv_channels)
            out_channels = self.shared_conv_channels[-1]
        else:
            out_channels = self.in_channels

        # cls specific branch
        prev_channel = out_channels
        if len(self.cls_conv_channels) > 0:
            self.cls_convs = self._add_conv_branch(prev_channel,
                                                   self.cls_conv_channels)
            prev_channel = self.cls_conv_channels[-1]
        self.conv_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=self.num_cls_out_channels,
            kernel_size=1)
        
        # reg specific branch
        prev_channel = out_channels
        if len(self.reg_conv_channels) > 0:
            self.reg_convs = self._add_conv_branch(prev_channel,
                                                   self.reg_conv_channels)
            prev_channel = self.reg_conv_channels[-1]
        self.conv_reg = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=self.num_reg_out_channels,
            kernel_size=1)
        
        # dir cls specific branch
        prev_channel = out_channels
        if len(self.dir_conv_channels):
            self.dir_convs = self._add_conv_branch(prev_channel,
                                                  self.dir_conv_channels)
            prev_channel = self.dir_conv_channels[-1]
        self.dir_cls = build_conv_layer(
            conv_cfg,
            in_channels=prev_channel,
            out_channels=self.num_dir_out_channels,
            kernel_size=1)

    def _add_conv_branch(self, in_channels, conv_channels):
        """Add shared or separable branch."""
        conv_spec = [in_channels] + list(conv_channels)
        # add branch specific conv layers
        conv_layers = nn.Sequential()
        for i in range(len(conv_spec) - 1):
            conv_layers.add_module(
                f'layer{i}',
                ConvModule(
                    conv_spec[i],
                    conv_spec[i + 1],
                    kernel_size=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.bias,
                    inplace=True))
        return conv_layers
        
    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = [TASK_UTILS.build(self.train_cfg.assigner)]
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]
            
    def forward(self, feats):
        """Forward.

        Args:
            feats (Tensor): Input features

        Returns:
            Tensor: Class scores predictions
            Tensor: Regression predictions
        """
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)
        else:
            x = feats

        # separate branches
        x_cls = x
        x_reg = x
        x_dir = x

        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)

        if len(self.reg_conv_channels) > 0:
            x_reg = self.reg_convs(x_reg)
        bbox_pred = self.conv_reg(x_reg)
        
        dir_score = None
        if self.with_dir_cls:
            if len(self.dir_conv_channels) > 0:
                x_dir = self.dir_convs(x_dir)
            dir_score = self.dir_cls(x_dir)

        return cls_score, bbox_pred, dir_score
            
    def predict(self,
                x: Tensor,
                batch_data_samples: SampleList,
                anchor_centroids: Tensor,
                test_cfg: ConfigType = None) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x Tensor: Frustum features of shape (N, C, L).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        test_cfg = self.test_cfg if test_cfg is None else test_cfg
        cls_scores, bbox_preds, dir_scores = self(x)
        
        priors = [
            self.prior_generator.grid_anchors(anchor_centroids[i], device=cls_scores.device)
            for i in range(x.shape[0])]
        
        results_list = []
        for i in range(bbox_preds.shape[0]):
            results = self._predict_single(
                cls_scores=cls_scores[i],
                bbox_preds=bbox_preds[i],
                dir_scores=dir_scores[i] if self.with_dir_cls else None,
                priors=priors[i],
                input_meta=batch_data_samples[i].metainfo,
                cfg=test_cfg)
            results_list.append(results)
        
        return results_list
    
    def _predict_single(self,
                        cls_scores: Tensor,
                        bbox_preds: Tensor,
                        dir_scores: Tensor,
                        priors: Tensor,
                        input_meta: dict,
                        cfg: ConfigDict) -> InstanceData:
        """Transform a single frustum points sample's features extracted from 
        the head into bbox results.

        Args:
            cls_scores (Tensor): Box scores for all frustum sections, 
                each item has shape (num_priors * num_classes, L).
            bbox_preds (Tensor): Box energies / deltas from
                all scale levels of a single point cloud sample, each item
                has shape (num_priors * C, L).
            priors (Tensor): The anchors. It has shape (L * num_priors, 7).
            input_meta (dict): Contain point clouds and image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = cls_scores.permute(1, 0).reshape(-1, self.num_classes)
        bbox_preds = bbox_preds.permute(1, 0).reshape(-1, self.box_code_size)
        
        dir_cls_score = None
        if self.with_dir_cls:
            dir_scores = dir_scores.permute(1, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_scores, dim=-1)[1]
        
        if self.use_sigmoid_cls:
            scores = cls_scores.sigmoid()
        else:
            scores = cls_scores.softmax(-1)
            
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            priors = priors[topk_inds, :]
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            if self.with_dir_cls:
                dir_cls_score = dir_cls_score[topk_inds]
            
        bbox_preds = self.bbox_coder.decode(priors, bbox_preds)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = scores.new_zeros(scores.shape[0], 1)
            scores = torch.cat([scores, padding], dim=1)
            
        bboxes_for_nms = input_meta['box_type_3d'](bbox_preds, box_dim=self.box_code_size)
        bboxes_for_nms = xywhr2xyxyr(bboxes_for_nms.bev)
            
        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(bbox_preds, bboxes_for_nms,
                                       scores, score_thr, cfg.max_num, cfg, dir_cls_score)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))

        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results
        
    def loss_and_targets(self, x: Tensor, batch_data_samples: SampleList, anchor_centroids: Tensor) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x Tensor: Frustum features of shape (N, C, L).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            dict: loss functions
        """
        cls_scores, bbox_preds, dir_scores = self(x)
        
        cls_scores = cls_scores.permute(0, 2, 1).reshape(-1, self.num_classes)
        bbox_preds = bbox_preds.permute(0, 2, 1).reshape(-1, self.bbox_coder.code_size)
        if self.with_dir_cls:
            dir_scores = dir_scores.permute(0, 2, 1).reshape(-1, 2)
        
        priors = [
            self.prior_generator.grid_anchors(anchor_centroids[i], device=cls_scores.device)
            for i in range(x.shape[0])]
            
        (labels, label_weights, bbox_targets, bbox_weights, 
         dir_targets, dir_weights, avg_factor) = self.get_targets(
            priors, batch_data_samples, self.train_cfg, concat=True)
        
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            priors = torch.cat(priors, dim=0)
            bbox_preds = self.bbox_coder.decode(priors, bbox_preds)
            bbox_preds = get_box_tensor(bbox_preds)
        
        losses_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=avg_factor)
        if self.diff_rad_by_sin:
            bbox_preds, bbox_targets = self.add_sin_difference(
                bbox_preds, bbox_targets)
        losses_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=avg_factor)
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        if self.with_dir_cls:
            loss_dir = self.loss_dir(
                dir_scores, dir_targets, dir_weights, avg_factor=avg_factor)
            losses['loss_dir'] = loss_dir
        return losses
    
    def get_targets(self,
                    priors: List[Tensor],
                    batch_data_samples: SampleList,
                    rcnn_train_cfg: ConfigDict = None,
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
        cfg = self.train_cfg if rcnn_train_cfg is None else rcnn_train_cfg
        
        (labels, label_weights, bbox_targets, bbox_weights, 
         dir_targets, dir_weights, avg_factors) = multi_apply(
            self._get_targets_single,
            priors,
            batch_data_samples,
            [cfg] * len(batch_data_samples))

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            dir_targets = torch.cat(dir_targets, 0)
            dir_weights = torch.cat(dir_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, sum(avg_factors)
    
    def _get_targets_single(self, priors: Tensor, data_samples: Det3DDataSample,
                            cfg: ConfigDict) -> tuple:
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_priors (Tensor): Contains all the positive boxes,
                has shape (num_pos, 7)
            neg_priors (Tensor): Contains all the negative boxes,
                has shape (num_neg, 7)
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 7)
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
        # match only with the x (depth) coordinate
        fake_priors = priors.clone()
        fake_priors[..., 1] = 0
        pred_instances_fake = InstanceData(priors=fake_priors)
        fake_gt_boxes = data_samples.gt_instances_3d.bboxes_3d.tensor.clone()
        fake_gt_boxes[..., 1] = 0
        gt_instances_fake = InstanceData(bboxes_3d=fake_gt_boxes, 
                                         labels_3d=data_samples.gt_instances_3d.labels_3d)
        
        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes_3d=data_samples.gt_instances_3d.bboxes_3d.tensor,
                                    labels_3d=data_samples.gt_instances_3d.labels_3d)

        assign_result = self.bbox_assigner.assign(
            pred_instances_fake, gt_instances_fake)
        sampling_result = self.bbox_sampler.sample(assign_result, pred_instances, gt_instances)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        num_pos = pos_inds.size(0)
        num_neg = neg_inds.size(0)
        num_samples = num_pos + num_neg
        
        # print((num_pos, num_neg, num_samples, data_samples.gt_instances_3d.bboxes_3d.tensor.shape))

        labels = priors.new_full((num_samples, ),
                                 self.num_classes,
                                 dtype=torch.long)
        reg_dim = self.bbox_coder.encode_size
        label_weights = priors.new_zeros(num_samples)
        bbox_targets = priors.new_zeros(num_samples, reg_dim)
        bbox_weights = priors.new_zeros(num_samples, reg_dim)
        dir_targets = priors.new_zeros(num_samples, dtype=torch.long)
        dir_weights = priors.new_zeros(num_samples)
        if num_pos > 0:
            labels[pos_inds] = sampling_result.pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = get_box_tensor(sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = pos_weight
            if self.with_dir_cls:
                dir_targets[pos_inds] = get_direction_target(
                    sampling_result.pos_priors,
                    pos_bbox_targets,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False,)
                dir_weights[pos_inds] = pos_weight
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, dir_targets, dir_weights, sampling_result.avg_factor
    
    @staticmethod
    def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> tuple:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2