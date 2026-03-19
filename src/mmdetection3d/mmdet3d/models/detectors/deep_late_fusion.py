import copy
import torch
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.structures import InstanceData
import random

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.det3d_data_sample import SampleList, OptSampleList


@MODELS.register_module()
class DeepLateFusion(Base3DDetector):
    
    # TODO: in the future, add the possibility to fine tune the img and lidar layers
    
    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 fusion_roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 valid_img_labels: Optional[List[int]] = None,
                 img_label_mapping: Optional[Dict[int, int]] = None,
                 proposals_2d_perturb: bool = False,
                 proposals_2d_perturb_std: Union[List[float], float] = 0.1,
                 proposals_3d_perturb: bool = False,
                 proposals_3d_perturb_std: Union[List[float], float] = 0.01,
                 perturb_proba: float = 0.3,
                 **kwargs):
        
        super(DeepLateFusion, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)
        
        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
        if pts_bbox_head:
            self.pts_train_cfg = train_cfg.pts_head if train_cfg else None
            pts_bbox_head.update(train_cfg=self.pts_train_cfg)
            self.pts_test_cfg = test_cfg.pts_head if test_cfg else None
            pts_bbox_head.update(test_cfg=self.pts_test_cfg)
            self.pts_bbox_head = MODELS.build(pts_bbox_head)

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_train_cfg = train_cfg.img_rpn if train_cfg is not None else None
            self.img_rpn_test_cfg = test_cfg.img_rpn
            img_rpn_head = img_rpn_head.copy()
            img_rpn_head.update(train_cfg=self.img_rpn_train_cfg, test_cfg=self.img_rpn_test_cfg)
            self.img_rpn_head = MODELS.build(img_rpn_head)
        if img_roi_head is not None:
            self.img_rcnn_train_cfg = train_cfg.img_rcnn if train_cfg is not None else None
            self.img_rcnn_test_cfg = test_cfg.img_rcnn
            img_roi_head.update(train_cfg=self.img_rcnn_train_cfg)
            img_roi_head.update(test_cfg=self.img_rcnn_test_cfg)
            self.img_roi_head = MODELS.build(img_roi_head)
            
        if fusion_roi_head is not None:
            self.fusion_roi_head = MODELS.build(fusion_roi_head)
            
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.valid_img_labels = torch.tensor(self.valid_img_labels) if valid_img_labels is not None else None
        # label_mapping, if passed, must be a dictionary containing all keys 
        # from 0 to the number of classes in the image detector
        if img_label_mapping is not None:
            self.img_label_mapping = torch.tensor([img_label_mapping[i] for i in range(len(img_label_mapping.keys()))])
        else:
            self.img_label_mapping = None
            
        self.proposals_2d_perturb = proposals_2d_perturb
        self.proposals_2d_perturb_std = proposals_2d_perturb_std  \
            if isinstance(proposals_3d_perturb_std, list) \
            else [proposals_2d_perturb_std] * 2
        self.proposals_3d_perturb = proposals_3d_perturb
        self.proposals_3d_perturb_std = proposals_3d_perturb_std \
            if isinstance(proposals_3d_perturb_std, list) \
            else [proposals_3d_perturb_std] * self.pts_bbox_head.box_code_size
        self.perturb_proba = perturb_proba
        
    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None
    
    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None
        
    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> Dict:
        """Extract features of images."""
        input_shape = img.shape[-2:]
        for img_meta in input_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats
    
    def extract_pts_feat(self, batch_inputs_dict: Dict[List, torch.Tensor]) -> Tuple[Tensor]:
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        pts_feats = self.extract_pts_feat(batch_inputs_dict)
        return (img_feats, pts_feats)
    
    def predict_pts(self, features: Tuple[Tensor], batch_data_samples: List[Det3DDataSample],
                    train: bool = True, **kwargs) -> List[InstanceData]:
        
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.pts_bbox_head(features)
        results_list = self.pts_bbox_head.predict_by_feat(
            *outs, 
            batch_input_metas=batch_input_metas, 
            cfg=self.pts_train_cfg if train else self.pts_test_cfg, 
            **kwargs)
        
        # results_list = self.pts_bbox_head.predict(features, batch_data_samples, **kwargs)
        return results_list
    
    def predict_imgs(self,
                     x: List[Tensor],
                     batch_data_samples: List[Det3DDataSample],
                     rescale: bool = True,
                     train: bool = True,
                     **kwargs) -> List[InstanceData]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            x (List[Tensor]): Image features from FPN.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.
        """

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.img_rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        batch_inputs_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        results_list = self.img_roi_head.predict_bbox(
            x, 
            batch_inputs_metas, 
            rpn_results_list, 
            rcnn_test_cfg=self.img_rcnn_train_cfg if train else self.img_rcnn_test_cfg, 
            rescale=rescale, 
            **kwargs)
        
        device = results_list[0].labels.device
        if self.valid_img_labels is not None:
            valid_labels = self.valid_img_labels.to(device)
        if self.img_label_mapping is not None:
            img_label_mapping = self.img_label_mapping.to(device)
        
        for result in results_list:
            if self.valid_img_labels is not None:
                valid_mask = torch.isin(result.labels, valid_labels)
                result.labels = result.labels[valid_mask]
                result.bboxes = result.bboxes[valid_mask]
                result.scores = result.scores[valid_mask]
            if self.img_label_mapping is not None:
                result.labels = img_label_mapping[result.labels]
            
        return results_list
    
    def loss_imgs(self, x: List[Tensor],
                  batch_data_samples: List[Det3DDataSample], **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances_3d.labels_3d)
            rpn_losses, rpn_results_list = self.img_rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)

        else:
            if 'proposals' in batch_data_samples[0]:
                # use pre-defined proposals in InstanceData
                # for the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            else:
                rpn_results_list = None
        roi_losses = self.img_roi_head.loss(x, rpn_results_list,
                                            batch_data_samples, **kwargs)
        losses.update(roi_losses)
        return losses
    
    def _forward(self,
                 batch_inputs_dict: Dict[List, torch.Tensor],
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        img_pred = self.predict_imgs(img_feats, batch_data_samples, **kwargs)
        pts_pred = self.predict_pts(pts_feats, batch_data_samples, **kwargs)
                
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, pts_pred, img_pred)
        
        return self.fusion_roi_head(img_feats, pts_feats, batch_data_samples, **kwargs)
    
    def loss(self, 
             batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                batch_input_metas)
        img_pred = self.predict_imgs(img_feats, batch_data_samples, train=True, **kwargs)
        # img_losses = self.loss_imgs(img_feats, batch_data_samples, **kwargs)
        
        # pts_losses, pts_pred = self.pts_bbox_head.loss_and_predict(pts_feats, batch_data_samples, **kwargs)
        # for key in pts_losses.keys():
        #     if 'loss' in key and 'pts' not in key:
        #         pts_losses[f'pts_{key}'] = pts_losses.pop(key)
        pts_pred = self.predict_pts(pts_feats, batch_data_samples, train=True, **kwargs)
        img_pred, pts_pred = self._perturb_bboxes(img_pred, pts_pred, batch_input_metas)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, pts_pred, img_pred)
        
        losses_refine = self.fusion_roi_head.loss(img_feats, pts_feats, batch_data_samples, **kwargs)
        # losses = dict()
        # losses.update(img_losses)
        # losses.update(pts_losses)
        # losses.update(losses_refine)
        return losses_refine
    
    def predict(self, 
                batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        img_pred = self.predict_imgs(img_feats, batch_data_samples, train=False, **kwargs)
        pts_pred = self.predict_pts(pts_feats, batch_data_samples, train=False, **kwargs)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, pts_pred, img_pred)
        
        results_3d = self.fusion_roi_head.predict(img_feats, pts_feats, batch_data_samples)
        detection_samples = self.add_pred_to_datasample(batch_data_samples, results_3d, img_pred)
        return detection_samples
    
    def _perturb_bboxes(self, 
                        proposals_2d: List[InstanceData], 
                        proposals_3d: List[InstanceData], 
                        batch_input_metas: List[Det3DDataSample]):
        
        if random.random() > self.perturb_proba:
            return proposals_2d, proposals_3d
        
        if self.proposals_2d_perturb:
            perturbed_proposals = []
            for proposals in proposals_2d: 
                ori_bboxes = proposals.bboxes
                noise_stds = ori_bboxes.new_tensor(
                    [self.proposals_2d_perturb_std]).repeat((1, 2)).repeat((ori_bboxes.shape[0], 1))
                new_proposal = InstanceData()
                new_proposal.bboxes = ori_bboxes + torch.normal(mean=0.0, std=noise_stds)
                new_proposal.labels = proposals.labels
                new_proposal.scores = proposals.scores
                perturbed_proposals.append(new_proposal)
        else:
            perturbed_proposals = proposals_2d
            
        if self.proposals_3d_perturb:
            perturbed_proposals_3d = []
            for proposals, metainfo in zip(proposals_3d, batch_input_metas): 
                ori_bboxes = proposals.bboxes_3d.tensor
                noise_stds = ori_bboxes.new_tensor(
                    [self.proposals_3d_perturb_std]).repeat((ori_bboxes.shape[0], 1))
                new_proposal = InstanceData()
                new_proposal.bboxes_3d = metainfo['box_type_3d'](ori_bboxes + torch.normal(mean=0.0, std=noise_stds))
                new_proposal.labels_3d = proposals.labels_3d
                new_proposal.scores_3d = proposals.scores_3d
                perturbed_proposals_3d.append(new_proposal)
        else:
            perturbed_proposals_3d = proposals_3d
            
        return perturbed_proposals, perturbed_proposals_3d
        