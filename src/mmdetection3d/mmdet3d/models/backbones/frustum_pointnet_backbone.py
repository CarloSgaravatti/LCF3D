from typing import Tuple, Sequence

import torch
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmdet3d.models.layers.pointnet_modules import build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from .base_pointnet import BasePointNet

ThreeTupleIntType = Tuple[Tuple[Tuple[int, int, int]]]
TwoTupleIntType = Tuple[Tuple[int, int, int]]
TwoTupleStrType = Tuple[Tuple[str]]


@MODELS.register_module()
class FrustumPointNetBackbone(BasePointNet):

    def __init__(self,
                 in_channels: int,
                 num_points: Tuple[int] = (2048, 1024, 512, 256),
                 radii: Tuple[Tuple[float, float, float]] = (
                     (0.2, 0.4, 0.8),
                     (0.4, 0.8, 1.6),
                     (1.6, 3.2, 4.8),
                 ),
                 num_samples: TwoTupleIntType = ((32, 32, 64), (32, 32, 64),
                                                 (32, 32, 32)),
                 sa_channels: ThreeTupleIntType = (((16, 16, 32), (16, 16, 32),
                                                    (32, 32, 64)),
                                                   ((64, 64, 128),
                                                    (64, 64, 128), (64, 96,
                                                                    128)),
                                                   ((128, 128, 256),
                                                    (128, 192, 256), (128, 256,
                                                                      256))),
                 aggregation_channels: Tuple[int] = (64, 128, 256),
                 fps_mods: TwoTupleStrType = (('D-FPS'), ('FS'), ('F-FPS',
                                                                  'D-FPS')),
                 fps_sample_range_lists: TwoTupleIntType = ((-1), (-1), (512,
                                                                         -1)),
                 dilated_group: Tuple[bool] = (True, True, True),
                 out_indices: Tuple[int] = (2, ),
                 norm_cfg: dict = dict(type='BN2d'),
                 sa_cfg: dict = dict(
                     type='PointSAModuleMSG',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=False),
                 bottleneck_mlp_channels: Sequence[int] = [128, 256, 1024],
                 bottleneck_norm_cfg: dict = dict(type='BN2d'),
                 bottleneck_sa_cfg: dict = dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_sa
        assert len(num_points) == len(radii) == len(num_samples) == len(
            sa_channels)
        if aggregation_channels is not None:
            assert len(sa_channels) == len(aggregation_channels)
        else:
            aggregation_channels = [None] * len(sa_channels)

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(radii[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            if isinstance(fps_mods[sa_index], tuple):
                cur_fps_mod = list(fps_mods[sa_index])
            else:
                cur_fps_mod = list([fps_mods[sa_index]])

            if isinstance(fps_sample_range_lists[sa_index], tuple):
                cur_fps_sample_range_list = list(
                    fps_sample_range_lists[sa_index])
            else:
                cur_fps_sample_range_list = list(
                    [fps_sample_range_lists[sa_index]])

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radii=radii[sa_index],
                    sample_nums=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    fps_mod=cur_fps_mod,
                    fps_sample_range_list=cur_fps_sample_range_list,
                    dilated_group=dilated_group[sa_index],
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    bias=True))
            skip_channel_list.append(sa_out_channel)

            cur_aggregation_channel = aggregation_channels[sa_index]
            if cur_aggregation_channel is None:
                self.aggregation_mlps.append(None)
                sa_in_channel = sa_out_channel
            else:
                self.aggregation_mlps.append(
                    ConvModule(
                        sa_out_channel,
                        cur_aggregation_channel,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=dict(type='BN1d'),
                        kernel_size=1,
                        bias=True))
                sa_in_channel = cur_aggregation_channel
            
            self.bottleneck_sa_ssg = build_sa_module(
                num_point=None,
                radius=None,
                num_sample=None,
                mlp_channels=bottleneck_mlp_channels,
                norm_cfg=bottleneck_norm_cfg,
                cfg=bottleneck_sa_cfg)

    def forward(self, points: Tensor):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            if self.aggregation_mlps[i] is not None:
                cur_features = self.aggregation_mlps[i](cur_features)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(sa_indices[-1])
                
        cur_xyz, cur_features, cur_indices = self.bottleneck_sa_ssg(
            sa_xyz[-1], sa_features[-1])
        out_sa_xyz.append(cur_xyz)
        out_sa_features.append(cur_features)
        out_sa_indices.append(
            torch.gather(sa_indices[-1], 1, cur_indices.long()))

        return dict(
            sa_xyz=out_sa_xyz,
            sa_features=out_sa_features,
            sa_indices=out_sa_indices)
