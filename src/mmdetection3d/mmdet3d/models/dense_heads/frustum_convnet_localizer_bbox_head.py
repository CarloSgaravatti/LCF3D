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
class FrustumConvNetLocalizerBboxHead(BaseModule):
    
    def __init__(self,
                 point_cloud_range: Sequence[float],
                 num_classes: int = 3,
                 anchor_sizes: Sequence[Sequence[int]] = [
                     [0.8, 0.6, 1.73],
                     [1.76, 0.6, 1.73], 
                     [3.9, 1.6, 1.56]],
                 rotations: Sequence[int] = [-3.14, -1.57, 0, 1.57],
                 in_channels=0,
                 shared_conv_channels=(),
                 cls_conv_channels=(),
                 reg_conv_channels=(),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 bias='auto',
                 init_cfg=None,
                 cls_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 center_loss: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 head_cls_loss: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 head_res_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 size_cls_loss: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 size_res_loss: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 gamma_corner: float = 0.1,
                 corner_loss: bool = True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 *args,
                 **kwargs):
        assert len(anchor_sizes) == num_classes
        
        super(FrustumConvNetLocalizerBboxHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)

        self.point_cloud_range = point_cloud_range
        self.anchor_sizes = anchor_sizes
        self.rotations = rotations
        self.num_anchors = len(self.anchor_sizes)
        self.num_head = len(self.rotations)
        
        self.use_sigmoid_cls = cls_loss.get('use_sigmoid', False)
        self.use_sigmoid_cls_size = size_cls_loss.get('use_sigmoid', False)
        self.use_sigmoid_cls_head = head_cls_loss.get('use_sigmoid', False)
        self.num_classes = num_classes
        self.total_num_classes = self.num_classes + 1 if not self.use_sigmoid_cls else self.num_classes
        self.num_reg_out_channels = 3 + 4 * self.num_anchors + 2 * self.num_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.cls_loss = MODELS.build(cls_loss)
        self.center_loss = MODELS.build(center_loss)
        self.head_cls_loss = MODELS.build(head_cls_loss)
        self.head_res_loss = MODELS.build(head_res_loss)
        self.size_cls_loss = MODELS.build(size_cls_loss)
        self.size_res_loss = MODELS.build(size_res_loss)
        self.corner_loss = corner_loss
        self.gamma_corner = gamma_corner
        
        self.in_channels = in_channels
        self.shared_conv_channels = shared_conv_channels
        self.cls_conv_channels = cls_conv_channels
        self.reg_conv_channels = reg_conv_channels
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
            out_channels=self.total_num_classes,
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
            
    def forward(self, feats):
        # shared part
        if len(self.shared_conv_channels) > 0:
            x = self.shared_convs(feats)
        else:
            x = feats

        # separate branches
        x_cls = x
        x_reg = x

        if len(self.cls_conv_channels) > 0:
            x_cls = self.cls_convs(x_cls)
        cls_score = self.conv_cls(x_cls)

        if len(self.reg_conv_channels) > 0:
            x_reg = self.reg_convs(x_reg)
        reg_output = self.conv_reg(x_reg)

        return cls_score, reg_output
            
    def predict(self,
                x: Tensor,
                batch_data_samples: SampleList) -> InstanceList:
        cls_score, reg_output = self(x)
        cls_score = cls_score.transpose(1, 2)
        reg_output = reg_output.transpose(1, 2)
        
        if self.use_sigmoid_cls:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)
            
        prior_sizes = cls_score.new_tensor(self.anchor_sizes).unsqueeze(0).unsqueeze(0)
        prior_sizes = prior_sizes.repeat(cls_score.shape[0], cls_score.shape[1], 1, 1)
        
        prior_head = cls_score.new_tensor(self.rotations).unsqueeze(0).unsqueeze(0)
        prior_head = prior_head.repeat(cls_score.shape[0], cls_score.shape[1], 1)
        
        featmap_size = cls_score.shape[1]
        frustum_section_shape = (self.point_cloud_range[3] - self.point_cloud_range[0]) / featmap_size
        x_centers = torch.linspace(
            self.point_cloud_range[0] + frustum_section_shape/2, 
            self.point_cloud_range[3] - frustum_section_shape/2,
            featmap_size, 
            device=cls_score.device).unsqueeze(1)
        y_centers = x_centers.new_zeros(size=x_centers.size())
        z_centers = x_centers.new_zeros(size=x_centers.size())
        centers = torch.cat([x_centers, y_centers, z_centers], dim=-1).unsqueeze(0).repeat(cls_score.shape[0], 1, 1)
        
        center_decoded = reg_output[:, :, :3] + centers
        size_decoded = reg_output[:, :, 3:3+self.num_anchors*3]
        size_decoded = size_decoded.view(-1, featmap_size, self.num_anchors, 3) * prior_sizes + prior_sizes
        size_cls = reg_output[:, :, 3+self.num_anchors*3:3+self.num_anchors*4]
        head_decoded = reg_output[:, :, 3+self.num_anchors*4:3+self.num_anchors*4+self.num_head]
        head_decoded = head_decoded.view(-1, featmap_size, self.num_head) * (np.pi/self.num_head) + prior_head
        head_cls = reg_output[:, :, 3+self.num_anchors*4+self.num_head:]
        
        batch_indices = torch.arange(cls_score.shape[0], device=cls_score.device)
        row_indices = torch.arange(cls_score.shape[1], device=cls_score.device)
        
        scores, labels = cls_score[..., :self.num_classes].max(dim=-1)
        scores_out, output_section_ids = scores.max(dim=-1)
        labels_out = labels[batch_indices, output_section_ids]
        
        center_decoded = center_decoded[batch_indices, output_section_ids, :]
        
        head_max = head_cls[batch_indices, output_section_ids].argmax(dim=-1)
        head_decoded = head_decoded[batch_indices, output_section_ids, head_max].unsqueeze(1)
        
        size_max = size_cls[batch_indices, output_section_ids].argmax(dim=-1)
        size_decoded = size_decoded.view(
            cls_score.shape[0], cls_score.shape[1], self.num_anchors, 3)
        size_decoded = size_decoded[batch_indices, output_section_ids, size_max, :]
        
        bboxes_out = torch.cat([
            center_decoded, size_decoded, head_decoded], dim=-1)
        
        return batch_data_samples, labels_out, bboxes_out, scores_out
        
    def loss_and_targets(self, x: Tensor, batch_data_samples: SampleList) -> InstanceList:
        box_type_3d = batch_data_samples[0].metainfo['box_type_3d']
        cls_scores, reg_outputs = self(x)
        cls_scores = cls_scores.transpose(1, 2).contiguous()
        reg_outputs = reg_outputs.transpose(1, 2).contiguous()
        
        batch_size = cls_scores.shape[0]
        featmap_size = cls_scores.shape[1]
        frustum_section_shape = (self.point_cloud_range[3] - self.point_cloud_range[0]) / featmap_size
        dummy_index_tensor = torch.arange(batch_size, device=reg_outputs.device)
        
        centers = reg_outputs[:, :, :3]
        sizes = reg_outputs[:, :, 3:3+self.num_anchors*3].view(-1, featmap_size, self.num_anchors, 3)
        size_cls = reg_outputs[:, :, 3+self.num_anchors*3:3+self.num_anchors*4]
        headings = reg_outputs[:, :, 3+self.num_anchors*4:3+self.num_anchors*4+self.num_head].view(-1, featmap_size, self.num_head)
        headings_cls = reg_outputs[:, :, 3+self.num_anchors*4+self.num_head:]
        
        if self.use_sigmoid_cls:
            cls_scores = cls_scores.sigmoid()
        else:
            cls_scores = cls_scores.softmax(-1)
        if self.use_sigmoid_cls_head:
            headings_cls = headings_cls.sigmoid()
        else:
            headings_cls = headings_cls.softmax(-1)
        if self.use_sigmoid_cls_size:
            size_cls = size_cls.sigmoid()
        else:
            size_cls = size_cls.softmax(-1)
            
        x_centers = torch.linspace(
            self.point_cloud_range[0] + frustum_section_shape/2, 
            self.point_cloud_range[3] - frustum_section_shape/2,
            featmap_size, 
            device=cls_scores.device).unsqueeze(1)
        y_centers = x_centers.new_zeros(size=x_centers.size())
        z_centers = x_centers.new_zeros(size=x_centers.size())
        prior_centers = torch.cat([x_centers, y_centers, z_centers], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
        
        prior_sizes = cls_scores.new_tensor(self.anchor_sizes).unsqueeze(0)
        prior_sizes = prior_sizes.repeat(batch_size, 1, 1)
        
        prior_head = cls_scores.new_tensor(self.rotations).unsqueeze(0)
        prior_head = prior_head.repeat(batch_size, 1)
        
        bboxes_gt = []
        corners_gt = []
        labels = []
        for i, data_sample in enumerate(batch_data_samples):
            label = data_sample.gt_instances_3d.labels_3d[0:1]
            bbox = data_sample.gt_instances_3d.bboxes_3d[0:1]
            bboxes_gt.append(bbox.tensor)
            corners_gt.append(bbox.corners)
            labels.append(label)
        bboxes_gt = torch.cat(bboxes_gt).contiguous()
        corners_gt = torch.cat(corners_gt).contiguous()
        corners_gt_flip = torch.cat([corners_gt[:, 4:, :], corners_gt[:, :4, :]], dim=1).contiguous()
        labels = torch.cat(labels).long().contiguous()
        
        center_gt = bboxes_gt[:, :3]
        size_gt = bboxes_gt[:, 3:6].unsqueeze(1).repeat(1, self.num_anchors, 1)
        heading_gt = bboxes_gt[:, 6:].repeat(1, self.num_head)
        heading_gt = limit_period(heading_gt, offset=0.5, period=2*np.pi)
        
        target_section = torch.floor(bboxes_gt[:, 0] / frustum_section_shape).long()
        
        center_targets = center_gt - prior_centers[dummy_index_tensor, target_section, :]
        # center_weights = center_targets.new_zeros((batch_size, featmap_size, 3))
        # center_weights[dummy_index_tensor, target_section, :] = 1
        # center_weights = center_weights
        
        anchor_reg_targets = (size_gt - prior_sizes) / prior_sizes
        anchor_reg_weights = anchor_reg_targets.new_zeros((batch_size, self.num_anchors, 3), dtype=torch.float32)
        anchor_reg_weights[dummy_index_tensor, labels, :] = 1
        anchor_reg_weights = anchor_reg_weights
        anchor_cls_targets = labels #.unsqueeze(1).repeat(1, featmap_size)
        # anchor_cls_weights = anchor_reg_targets.new_zeros((batch_size, featmap_size), dtype=torch.float32)
        # anchor_cls_weights[dummy_index_tensor, target_section] = 1
        # anchor_cls_weights = anchor_cls_weights
                
        heading_reg_targets = (heading_gt - prior_head) / (np.pi/self.num_head)
        heading_reg_targets = heading_reg_targets
        heading_reg_weights = anchor_reg_targets.new_zeros((batch_size, self.num_head), dtype=torch.float32)   
        angle_per_class = 2*np.pi / float(self.num_head)
        # summing since angles are between -np.pi and np.pi
        heading_gt_cls = (bboxes_gt[:, 6] + np.pi + (angle_per_class / 2)) % (2*np.pi)
        heading_gt_cls = (heading_gt_cls / angle_per_class).long()
        heading_reg_weights[dummy_index_tensor, heading_gt_cls] = 1
        # heading_reg_weights = heading_reg_weights
        heading_cls_targets = heading_gt_cls #.unsqueeze(1).repeat(1, featmap_size)
        # heading_cls_weights = anchor_reg_targets.new_zeros((batch_size, featmap_size), dtype=torch.float32)
        # heading_cls_weights[dummy_index_tensor, target_section] = 1
        # heading_cls_weights = heading_cls_weights
        
        cls_target = anchor_reg_targets.new_full((batch_size, featmap_size), self.num_classes, dtype=torch.long)
        cls_target[dummy_index_tensor, target_section] = labels

        loss_cls = self.cls_loss(cls_scores.transpose(1, 2), cls_target)
        loss_center = self.center_loss(centers[dummy_index_tensor, target_section, :], center_targets) #, center_weights)
        loss_size_res = self.size_res_loss(sizes[dummy_index_tensor, target_section, :, :], 
                                           anchor_reg_targets, anchor_reg_weights)
        loss_size_cls = self.size_cls_loss(size_cls[dummy_index_tensor, target_section, :], 
                                           anchor_cls_targets) #, anchor_cls_weights)
        loss_head_res = self.head_res_loss(headings[dummy_index_tensor, target_section, :], 
                                           heading_reg_targets, heading_reg_weights)
        loss_head_cls = self.head_cls_loss(headings_cls[dummy_index_tensor, target_section, :], 
                                           heading_cls_targets) #, heading_cls_weights)
        losses = dict(loss_cls=loss_cls,loss_center=loss_center, loss_size_res=loss_size_res,
                      loss_size_cls=loss_size_cls, loss_head_res=loss_head_res, loss_head_cls=loss_head_cls)
        if self.corner_loss:
            bboxes_pred = torch.cat([
                centers[dummy_index_tensor, target_section, :] + \
                    prior_centers[dummy_index_tensor, target_section, :],
                sizes[dummy_index_tensor, target_section, labels, :] + \
                    prior_sizes[dummy_index_tensor, labels, :],
                headings[dummy_index_tensor, target_section, heading_gt_cls].unsqueeze(1) + \
                    prior_head[dummy_index_tensor, heading_gt_cls].unsqueeze(1)
            ], dim=1).contiguous()
            corners_pred = box_type_3d(bboxes_pred).corners
            loss_corner = torch.sqrt(torch.sum((corners_pred - corners_gt)**2, dim=-1)).sum(dim=-1)
            # loss_corner = self.corner_crit(corners_pred, corners_gt).sum(dim=-1)
            loss_corner_flip = torch.sqrt(torch.sum((corners_pred - corners_gt_flip)**2, dim=-1)).sum(dim=-1)
            loss_corner = torch.minimum(loss_corner, loss_corner_flip)
            loss_corner = loss_corner.mean() * self.gamma_corner
            losses.update({'loss_corner': loss_corner})
        return losses