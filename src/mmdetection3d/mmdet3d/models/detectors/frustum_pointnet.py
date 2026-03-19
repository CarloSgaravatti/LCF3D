from typing import Tuple, Sequence, Optional
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from .base import Base3DDetector
from mmdet3d.structures import limit_period, xywhr2xyxyr


class TNet(nn.Module):
    def __init__(self, k: int, with_one_hot: int = 0):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256 + with_one_hot, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)      

    def forward(self, x, one_hot=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        
        if one_hot is not None:
            x = torch.cat([x, one_hot], dim=-1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


@MODELS.register_module()
class FrustumPointNet(Base3DDetector):

    def __init__(self,
                 seg_backbone: ConfigType,
                 seg_decode_head: ConfigType,
                 box_estimation_backbone: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 fc_input_channels: int = 128,
                 fc_channels: Sequence[int] = [512, 256],
                 anchors: Sequence[Sequence[int]] = [
                     [0.8, 0.6, 1.73], 
                     [1.76, 0.6, 1.73], 
                     [3.9, 1.6, 1.56]],
                 rotations: Sequence[int] = [-3.14, -2.617, -2.093, -1.57, -1.046, -0.523,
                                             0, 0.523, 1.046, 1.57, 2.093, 2.617],
                 loss_center_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 loss_center_tnet: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 loss_cls_anchors: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_anchors_reg: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 loss_cls_headings: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_headings_reg: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=1.0),
                 max_num_points: int = 256,
                 min_num_points_seg: int = 16,
                 max_num_points_seg: int = 128,
                 final_dropout: float = 0.2,
                 with_one_hot: bool = False,
                 one_hot_len: int = 3,
                 gamma_corner: float = 0.1,
                 corner_loss: bool = True,
                 init_cfg: OptMultiConfig = None,
                 train_cfg: dict = dict(sample_seg_points=128),
                 test_cfg: dict = dict(sample_seg_points=128),) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.seg_backbone = MODELS.build(seg_backbone)
        self.seg_decode_head = MODELS.build(seg_decode_head)
        self.box_estimation_backbone = MODELS.build(box_estimation_backbone)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_num_points_seg = max_num_points_seg
        self.min_num_points_seg = min_num_points_seg
        self.with_one_hot = with_one_hot
        self.one_hot_len = one_hot_len
        self.max_num_points = max_num_points
        self.final_dropout = final_dropout
        
        self.use_sigmoid_anchor_cls = loss_cls_anchors.get('use_sigmoid', False)
        self.use_sigmoid_heading_cls = loss_cls_headings.get('use_sigmoid', False)
        self.loss_center_bbox = MODELS.build(loss_center_bbox)
        self.loss_center_tnet = MODELS.build(loss_center_tnet)
        self.loss_cls_anchors = MODELS.build(loss_cls_anchors)
        self.loss_anchors_reg = MODELS.build(loss_anchors_reg)
        self.loss_cls_headings = MODELS.build(loss_cls_headings)
        self.loss_headings_reg = MODELS.build(loss_headings_reg)
        self.gamma_corner = gamma_corner
        self.corner_loss = corner_loss
        if self.corner_loss:
            self.corner_crit = torch.nn.L1Loss(reduction='none')
            
        self.tnet = TNet(3, 0 if not self.with_one_hot else self.one_hot_len)
        
        self.anchors = anchors
        self.rotations = rotations
        self.num_anchor_size = len(anchors)
        self.num_headings = len(rotations)
        self.fc_layers = []
        self.bn_layers = []
        input_channels = fc_input_channels
        for channels in fc_channels:
            self.fc_layers.append(nn.Linear(input_channels, channels))
            self.bn_layers.append(nn.BatchNorm1d(channels))
            input_channels = channels
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)
        self.output_fc = nn.Linear(input_channels, 3+self.num_anchor_size*4+self.num_headings*2)

    def extract_feat(self, batch_points: Tensor, batch_one_hot: Tensor = None) -> Tuple[Tensor]:
        x = self.seg_backbone(batch_points)
        sa_xyz, sa_features = self.seg_decode_head._extract_input(x)
        sa_features[0] = None
        fp_feature = sa_features[-1]

        if batch_one_hot is not None:
            fp_feature = torch.cat([fp_feature, batch_one_hot.unsqueeze(2)], dim=1)

        for i in range(self.seg_decode_head.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.seg_decode_head.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                                            sa_features[-(i + 2)], fp_feature)
        seg_output = self.seg_decode_head.pre_seg_conv(fp_feature)
        seg_output = self.seg_decode_head.cls_seg(seg_output)

        return seg_output, fp_feature
    
    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None,
                 train: bool = False,
                 **kwargs):
        # points = torch.stack(batch_inputs_dict['points'])
        
        # lengths = [p.shape[0] for p in batch_inputs_dict['points']]
        # min_l, max_l = min(lengths), max(lengths)
        # num_initial_points = min_l if min_l > self.max_num_points else max_l
        # num_initial_points = min(max([p.shape[0] for p in batch_inputs_dict['points']]), self.max_num_points)
        num_initial_points = min(max([p.shape[0] for p in batch_inputs_dict['points']]), self.max_num_points_seg)
        points = []
        sampled_indices = []
        for pc in batch_inputs_dict['points']:
            n_points = pc.shape[0]
            if n_points > num_initial_points:
                choices = np.random.choice(n_points,
                    num_initial_points, replace=False)
            else:
                choices = np.random.choice(n_points,
                    num_initial_points-n_points, replace=True)
                choices = np.concatenate((np.arange(n_points), choices))
            choices = pc.new_tensor(choices, dtype=torch.int32)
            points.append(pc[choices, :])
            sampled_indices.append(choices)
        points = torch.stack(points)
        
        one_hot = None
        if self.with_one_hot:
            one_hot = torch.stack([ds.metainfo['one_hot_vector'] for ds in batch_data_samples]).to(points.device)
            
        seg_logits, fp_feature = self.extract_feat(points, one_hot)
        mask = torch.argmax(seg_logits.transpose(1, 2), dim=-1)
        num_segmented_points = mask.sum(dim=1, keepdim=True)
        # features = fp_feature.transpose(1, 2)
        
        seg_center = (points[..., :3] * mask.unsqueeze(2)).sum(dim=1) / (num_segmented_points + 1e-4)
        
        batch_seg_points = []
        # batch_seg_features = []
        # min_s, max_s = num_segmented_points.squeeze(1).min().item(), num_segmented_points.squeeze(1).max().item()
        # num_points_seg = min_s if min_s > self.max_num_points_seg else max_s
        # num_points_seg = max(num_points_seg, self.min_num_points_seg)
        num_points_seg = max(self.min_num_points_seg, min(num_segmented_points.squeeze(1).max().item(), self.max_num_points_seg))
        for i in range(points.shape[0]):
            pos_indices = torch.where(mask[i, :] > 0.5)[0]
            npositive = pos_indices.shape[0]
            if npositive > 0:
                if npositive > num_points_seg:
                    choices = np.random.choice(npositive,
                        num_points_seg, replace=False)
                else:
                    choices = np.random.choice(npositive,
                        num_points_seg-npositive, replace=True)
                    choices = np.concatenate((np.arange(npositive), choices))
                choices = pos_indices[points.new_tensor(choices, dtype=torch.int32)]
            else:
                choices = points.new_zeros((num_points_seg,), dtype=torch.int32)
            batch_seg_points.append(points[i, choices, :])
            # batch_seg_features.append(features[i, choices, :])
            
        seg_points = torch.stack(batch_seg_points).contiguous()
        # seg_features = torch.stack(batch_seg_features).contiguous()
        
        points_seg_coords = seg_points[..., :3] - seg_center.unsqueeze(1)
        tnet_center = self.tnet(points_seg_coords[..., :3].transpose(1, 2).contiguous(), one_hot)
        
        bbox_est_input = torch.cat([points_seg_coords - tnet_center.unsqueeze(1), seg_points[..., 3:]], dim=-1).contiguous()
        bbox_features = self.box_estimation_backbone(bbox_est_input)['sa_features'][-1].squeeze(2)
        if self.with_one_hot:
            bbox_features = torch.cat([bbox_features, one_hot], dim=-1)
        for i in range(len(self.fc_layers)):
            bbox_features = self.fc_layers[i](bbox_features)
            bbox_features = self.bn_layers[i](bbox_features)
            bbox_features = F.relu(bbox_features)
        outputs = self.output_fc(bbox_features)
        output_center_residual = outputs[..., :3]
        output_sizes_reg = outputs[..., 3:3+self.num_anchor_size*3]
        output_sizes_cls = outputs[..., 3+self.num_anchor_size*3:3+self.num_anchor_size*4]
        output_headings_reg = outputs[..., 3+self.num_anchor_size*4:3+self.num_anchor_size*4+self.num_headings]
        output_headings_cls = outputs[..., 3+self.num_anchor_size*4+self.num_headings:]
        
        return (sampled_indices, one_hot, seg_logits, seg_center, tnet_center, output_center_residual, 
                output_sizes_reg, output_sizes_cls, output_headings_reg, output_headings_cls)
    
    def loss(self,
             batch_inputs_dict: dict,
             batch_data_samples: SampleList,
             **kwargs):
        box_type_3d = batch_data_samples[0].metainfo['box_type_3d']
                
        (sampled_indices, one_hot, seg_logits, seg_center, tnet_center, output_center_residual, output_sizes_reg, 
         output_sizes_cls, output_headings_reg, output_headings_cls) = self._forward(batch_inputs_dict, batch_data_samples, train=True)
        output_sizes_reg = output_sizes_reg.view(-1, self.num_anchor_size, 3)
        
        if self.use_sigmoid_anchor_cls:
            output_sizes_cls = output_sizes_cls.sigmoid()
        else:
            output_sizes_cls = output_sizes_cls.softmax(-1)
        if self.use_sigmoid_heading_cls:
            output_headings_cls = output_headings_cls.sigmoid()
        else:
            output_headings_cls = output_headings_cls.softmax(-1)
        
        batch_size = seg_logits.shape[0]
        dummy_index_tensor = torch.arange(batch_size, device=seg_logits.device)
        
        seg_targets = []
        bboxes_gt = []
        corners_gt = []
        anchor_cls_targets = []
        for i, data_sample in enumerate(batch_data_samples):
            # labels used for size regression anchor selection for targets
            label = data_sample.gt_instances_3d.labels_3d[0:1]
            bbox = data_sample.gt_instances_3d.bboxes_3d[0:1]
            seg_targets.append(data_sample.gt_pts_seg.pts_semantic_mask[sampled_indices[i]])
            bboxes_gt.append(bbox.tensor)
            corners_gt.append(bbox.corners)
            anchor_cls_targets.append(label)
        seg_targets = torch.stack(seg_targets).contiguous().long().to(seg_logits.device)
        bboxes_gt = torch.cat(bboxes_gt).contiguous()
        corners_gt = torch.cat(corners_gt).contiguous()
        anchor_cls_targets = torch.cat(anchor_cls_targets).contiguous()
        # flipping means that the set of corners is exactly the same
        # in mmdet3d, the first 4 corners are in the front facet (considering the front
        # based on the yaw angle), so to retrieve the flipped corners it is sufficient
        # to exchange the two set of corners
        corners_gt_flip = torch.cat([corners_gt[:, 4:, :], corners_gt[:, :4, :]], dim=1).contiguous()
        
        center_gt = bboxes_gt[:, :3]
        size_gt = bboxes_gt[:, 3:6].unsqueeze(1).repeat(1, self.num_anchor_size, 1)
        heading_gt = bboxes_gt[:, 6:].repeat(1, self.num_headings)
        heading_gt = limit_period(heading_gt, offset=0.5, period=2*np.pi)
        
        anchor_sizes = seg_logits.new_tensor(self.anchors).unsqueeze(0).repeat(batch_size, 1, 1)
        headings = seg_logits.new_tensor(self.rotations).unsqueeze(0).repeat(batch_size, 1)
        
        anchor_reg_targets = (size_gt - anchor_sizes) / anchor_sizes
        anchor_reg_weights = anchor_reg_targets.new_zeros((batch_size, self.num_anchor_size, 3), dtype=torch.float32)
        anchor_reg_weights[dummy_index_tensor, anchor_cls_targets, :] = 1
        
        heading_reg_targets = (heading_gt - headings) / (np.pi/self.num_headings)
        heading_reg_weights = anchor_reg_targets.new_zeros((batch_size, self.num_headings), dtype=torch.float32)   
        angle_per_class = 2*np.pi / float(self.num_headings)
        # summing since angles are between -np.pi and np.pi
        heading_cls_targets = (heading_gt[:, 0] + np.pi + (angle_per_class / 2)) % (2*np.pi)
        heading_cls_targets = (heading_cls_targets / angle_per_class).long()
        heading_reg_weights[dummy_index_tensor, heading_cls_targets] = 1

        loss_seg = self.seg_decode_head.loss_decode(
            seg_logits, seg_targets, ignore_index=self.seg_decode_head.ignore_index)
        loss_tnet = self.loss_center_tnet(tnet_center + seg_center, center_gt)
        loss_center = self.loss_center_bbox(tnet_center + seg_center + output_center_residual, center_gt)
        loss_size_reg = self.loss_anchors_reg(
            output_sizes_reg, anchor_reg_targets, anchor_reg_weights, avg_factor=anchor_reg_weights.sum())
        loss_size_cls = self.loss_cls_anchors(output_sizes_cls, anchor_cls_targets)
        loss_heading_reg = self.loss_headings_reg(output_headings_reg, heading_reg_targets, 
                                                  heading_reg_weights, avg_factor=heading_reg_weights.sum())
        loss_heading_cls = self.loss_cls_headings(output_headings_cls, heading_cls_targets)
        losses = dict(
            loss_seg=loss_seg, loss_tnet=loss_tnet, loss_center=loss_center, loss_size_reg=loss_size_reg,
            loss_size_cls=loss_size_cls, loss_heading_reg=loss_heading_reg, loss_heading_cls=loss_heading_cls)
        
        if self.corner_loss:
            bboxes_pred = torch.cat([
                seg_center + tnet_center + output_center_residual,
                anchor_sizes[dummy_index_tensor, anchor_cls_targets, :] + \
                    output_sizes_reg[dummy_index_tensor, anchor_cls_targets, :],
                headings[dummy_index_tensor, heading_cls_targets].unsqueeze(1) + \
                    output_headings_reg[dummy_index_tensor, heading_cls_targets].unsqueeze(1)
            ], dim=1).contiguous()
            corners_pred = box_type_3d(bboxes_pred).corners
            loss_corner = torch.sqrt(torch.sum((corners_pred - corners_gt)**2, dim=-1)).sum(dim=-1)
            # loss_corner = self.corner_crit(corners_pred, corners_gt).sum(dim=-1)
            loss_corner_flip = torch.sqrt(torch.sum((corners_pred - corners_gt_flip)**2, dim=-1)).sum(dim=-1)
            loss_corner = torch.minimum(loss_corner, loss_corner_flip)
            loss_corner = loss_corner.mean() * self.gamma_corner
            losses.update({'loss_corner': loss_corner})
        
        return losses
        
    def predict(self, 
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                **kwargs):
        (sampled_indices, one_hot, seg_logits, seg_center, tnet_center, output_center_residual, output_sizes_reg, 
         output_sizes_cls, output_headings_reg, output_headings_cls) = self._forward(batch_inputs_dict, batch_data_samples, train=False)
        
        batch_size = seg_logits.shape[0]
        dummy_index_tensor = torch.arange(batch_size, device=seg_logits.device)
        anchor_sizes = seg_logits.new_tensor(self.anchors).unsqueeze(0).repeat(batch_size, 1, 1)
        headings = seg_logits.new_tensor(self.rotations).unsqueeze(0).repeat(batch_size, 1)
        
        # de-normalize output regressions
        output_headings_reg *= (np.pi/self.num_headings)
        output_sizes_reg = output_sizes_reg.view(-1, self.num_anchor_size, 3) * anchor_sizes
            
        center = seg_center + tnet_center + output_center_residual
        size_ids = output_sizes_cls.argmax(dim=-1) #.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3)
        output_sizes_reg = output_sizes_reg + anchor_sizes
        sizes = output_sizes_reg[dummy_index_tensor, size_ids, :] #torch.gather(output_sizes_reg, dim=1, index=size_ids)
        heading_ids = output_headings_cls.argmax(dim=-1) #.unsqueeze(1)
        output_headings_reg += headings
        heading = output_headings_reg[dummy_index_tensor, heading_ids] #torch.gather(output_headings_reg, dim=-1, index=heading_ids)
        heading = limit_period(heading, offset=0.5, period=2*np.pi)
        bboxes = torch.cat([center, sizes, heading.unsqueeze(1)], dim=1)
            
        return batch_data_samples, bboxes