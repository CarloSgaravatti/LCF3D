from typing import List, Tuple, Union, Generator
import torch
from torch import Tensor
import numpy as np

from mmdet3d.registry import TASK_UTILS

@TASK_UTILS.register_module()
class FrustumAnchorGenerator(object):
    """
    Anchor generator for Frustum ConvNet like networks

    Args:
        frustum_range (list[float]): Range of the frustum in the format
            [x_min, y_min, z_min, x_max, y_max, z_max].
        num_slices (int): Number of slices to divide the frustum into.
        anchor_sizes (list[list[float]]): Sizes of the anchors.
        height (float): Height at which to place the anchors.
        rotations (list[float], optional): List of rotations for the anchors.
            Defaults to [0, 1.5707963].
        custom_values (tuple[float], optional): Custom values for the anchors.
            Defaults to ().
        reshape_out (bool, optional): Whether to reshape the output anchors.
            Defaults to True.
    """
    def __init__(self,
                 frustum_range: List[float],
                 anchor_sizes: List[List[float]],
                 heights: float,
                 rotations: List[float] = [0, 1.5707963]):
        self.frustum_range = frustum_range
        self.anchor_sizes = anchor_sizes
        self.heights = heights
        self.rotations = rotations
        self.num_base_anchors = len(self.rotations) * len(self.anchor_sizes)

    def grid_anchors(self, frustum_centroids: Tensor, device: Union[str, torch.device] = 'cuda'):
        """
        Generate grid anchors for all frustum slices.

        Args:
            featmap_size (int): Size of the feature map.
            device (str or torch.device, optional): Device on which to create the anchors.
                Defaults to 'cuda'.

        Returns:
            list[torch.Tensor]: List of tensors containing the anchors for each slice.
        """
        featmap_size = frustum_centroids.shape[0]
        frustum_section_shape = (self.frustum_range[3] - self.frustum_range[0]) / featmap_size
        
        class_anchors = []
        for i in range(len(self.anchor_sizes)):
            
            x_centers = torch.linspace(
                self.frustum_range[0] + frustum_section_shape/2, 
                self.frustum_range[3] - frustum_section_shape/2,
                featmap_size, 
                device=device)
            y_centers = x_centers.new_zeros(size=x_centers.size())
            z_centers = x_centers.new_ones(size=x_centers.size()) * self.heights[i]
            
            anchor_centers = torch.cat([
                x_centers.unsqueeze(1),
                y_centers.unsqueeze(1),
                z_centers.unsqueeze(1)], dim=1)
            anchor_centers = torch.repeat_interleave(
                anchor_centers, repeats=len(self.rotations), dim=0)
            
            anchor_dimensions = torch.repeat_interleave(
                anchor_centers.new_tensor(self.anchor_sizes[i]).unsqueeze(0), repeats=len(self.rotations), dim=0)
            anchor_rotations = anchor_centers.new_tensor(self.rotations)
            anchors_sizes = torch.cat([anchor_dimensions, anchor_rotations.unsqueeze(1)], dim=1)
            anchors_sizes = anchors_sizes.repeat(featmap_size, 1)
            
            anchors = torch.cat([anchor_centers, anchors_sizes], dim=1)
            class_anchors.append(anchors)
            
        return class_anchors
        
        # x_centers = torch.linspace(
        #     self.frustum_range[0] + frustum_section_shape/2, 
        #     self.frustum_range[3] - frustum_section_shape/2,
        #     featmap_size, 
        #     device=device)
        # y_centers = x_centers.new_zeros(size=x_centers.size())
        # # z_centers = x_centers.new_ones(size=x_centers.size()) * self.height
        
        # anchor_centers = torch.cat([
        #     x_centers.unsqueeze(1),
        #     y_centers.unsqueeze(1)], dim=1)
        #     # z_centers.unsqueeze(1)], dim=1)
        # anchor_centers = torch.repeat_interleave(
        #     anchor_centers, repeats=self.num_base_anchors, dim=0) #frustum_centroids
        
        # heights = torch.repeat_interleave(
        #     anchor_centers.new_tensor(self.heights), repeats=len(self.rotations), dim=0)
        # anchor_dimensions = torch.repeat_interleave(
        #     anchor_centers.new_tensor(self.anchor_sizes), repeats=len(self.rotations), dim=0)
        # anchor_rotations = anchor_centers.new_tensor(self.rotations).repeat(len(self.anchor_sizes))
        
        # anchors_sizes = torch.cat([heights.unsqueeze(1), anchor_dimensions, anchor_rotations.unsqueeze(1)], dim=1)
        # anchors_sizes = anchors_sizes.repeat(featmap_size, 1)
        
        # anchors = torch.cat([anchor_centers, anchors_sizes], dim=1)
        # return anchors.to(device)