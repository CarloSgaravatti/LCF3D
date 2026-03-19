from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import get_box_type
from typing_extensions import Union
import numpy as np
import torch


@DATASETS.register_module()
class FrustumDataset(Det3DDataset):
    
    METAINFO = {
        'classes': ['Pedestrian', 'Cyclist', 'Car', 'Background']
    }
    
    def __init__(self, metainfo=None, **kwargs):
        super().__init__(metainfo=metainfo, **kwargs)
        
    def parse_data_info(self, info: dict) -> dict:
        keys = ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 
                'gt_bboxes_left', 'gt_bboxes_right', 'pts_semantic_mask',
                'lidar_to_cam', 'cam_to_img', 'cam_to_lidar', 'one_hot_vector']
        dtypes = {
            'gt_bboxes_3d': torch.float32,
            'gt_labels_3d': torch.long,
            'gt_bboxes': torch.float32,
            'gt_bboxes_left': torch.float32,
            'gt_bboxes_right': torch.float32,
            'gt_labels': torch.long,
            'pts_semantic_mask': torch.float32,
            'lidar_to_cam': torch.float32,
            'cam_to_img': torch.float32,
            'cam_to_lidar': torch.float32,
            'one_hot_vector': torch.float32,
        }
        for key in keys:
            if key in info:
                info[key] = torch.tensor(np.array(info[key]), dtype=dtypes[key])
        info['gt_bboxes_3d'] = self.box_type_3d(info['gt_bboxes_3d'].squeeze(1))
        
        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
    
    def parse_ann_info(self, info: dict) -> Union[dict, None]:
        keys = ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'pts_semantic_mask']
        
        ann_info = dict()
        for key in keys:
            if key in info:
                ann_info[key] = info[key]
                
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                self.num_ins_per_cat[label] += 1
                    
        return ann_info