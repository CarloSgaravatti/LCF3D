import torch
from torch import Tensor
from typing_extensions import List, Tuple, Optional, Dict
from mmdet.utils import ConfigType, OptMultiConfig

from mmdet3d.models.roi_heads.roi_extractors import SingleRoIExtractor
from mmdet3d.structures.bbox_3d import xywhr2xyxyr
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AlignedBEVRoIExtractor(SingleRoIExtractor):
    
    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 grid_size: Tuple[int, int],
                 point_cloud_range: Tuple,
                 finest_scale: int = 56,
                 init_cfg: OptMultiConfig = None):
        super(AlignedBEVRoIExtractor, self).__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            finest_scale=finest_scale,
            init_cfg=init_cfg)
        
        mult = [-1, 1]
        self.corners_multipliers = torch.tensor([[x_mult, y_mult] for x_mult in mult for y_mult in mult], dtype=torch.float32)
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        
    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None):
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 6) where the first
                column indicates batch id of each RoI. The other 5 columns
                are the bev bounding boxes, defined as (x,y,w,h,r)
            roi_scale_factor (Optional[float]): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        rois = self._convert_to_grid_coords(rois)
        return super(AlignedBEVRoIExtractor, self).forward(feats, rois, roi_scale_factor)
        
    def _convert_to_xyxy(self, rois: Tensor):
        '''From (batch_id,x,y,w,h,yaw) to (batch_id,x1,y1,x2,y2)'''
        half_dimensions = rois[:, 3:5].reshape(-1, 1, 2).repeat(1, 4, 1) / 2
        centers = rois[:, 1:3].reshape(-1, 1, 2).repeat(1, 4, 1)
        
        # directions for the yaw rotation
        direction_1 = torch.cos(rois[:, -1])
        direction_2 = torch.sin(rois[:, -1])
    
        corners_relative = self.corners_multipliers.to(rois.device) * half_dimensions
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
        
        return torch.stack((rois[:, 0], min_x, min_y, max_x, max_y), dim=1)
    
    def _convert_to_grid_coords(self, rois: Tensor):
        '''
        Convert RoIs from real-world coordinates (batch_id,x,y,w,h,yaw) 
        to BEV image grid coordinates (batch_id,x1,y1,x2,y2).
        '''
        rois = self._convert_to_xyxy(rois)
        
        rois[:, 1] = (rois[:, 1] - self.point_cloud_range[0]) / self.grid_size[0]
        rois[:, 2] = (rois[:, 2] - self.point_cloud_range[1]) / self.grid_size[1]
        rois[:, 3] = (rois[:, 3] - self.point_cloud_range[0]) / self.grid_size[0]
        rois[:, 4] = (rois[:, 4] - self.point_cloud_range[1]) / self.grid_size[1]
        
        return rois