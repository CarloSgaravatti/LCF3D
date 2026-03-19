_base_ = ['../_base_/schedules/cyclic-40e.py']


##########################################################################
##########################   NETWORK SETTINGS   ##########################
##########################################################################
voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
pointpillars_path = '/mnt/proj2/dd-24-8/torch_checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
faster_rcnn_path = '/mnt/proj2/dd-24-8/mmdetection_work_dir/base_faster_rcnn_fpn_3class/best_coco_bbox_mAP_epoch_200.pth'

model = dict(
    type='DeepLateFusion',
    img_label_mapping={0: 2, 1: 0, 2: 1},
    proposals_2d_perturb=True,
    proposals_2d_perturb_std=[2, 1],
    proposals_3d_perturb=True,
    proposals_3d_perturb_std=[0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        # PointPillars
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            voxel_size=voxel_size,
            max_voxels=(16000, 40000)),
        # Faster RCNN
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_size_divisor=32,
        std=[58.395, 57.12, 57.375],),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', 
        in_channels=64, 
        output_shape=[496, 432]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pointpillars_path,
            prefix='backbone',
        )),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pointpillars_path,
            prefix='neck',
        )),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pointpillars_path,
            prefix='bbox_head',
        )),
    img_backbone=dict(
        depth=101,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3,),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=faster_rcnn_path,
            prefix='backbone',
        ),
        type='mmdet.ResNet'),
    img_neck=dict(
        in_channels=[256, 512, 1024, 2048],
        num_outs=5,
        out_channels=256,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=faster_rcnn_path,
            prefix='neck',
        ),
        type='mmdet.FPN'),
    img_roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                type='mmdet.DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='mmdet.L1Loss'),
            loss_cls=dict(loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=False),
            num_classes=3,
            reg_class_agnostic=False,
            roi_feat_size=7,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            type='mmdet.Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='mmdet.SingleRoIExtractor'),
        test_cfg=dict(
            max_per_img=1000,
            nms=dict(iou_threshold=0.05, type='nms'),
            score_thr=0.05),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=faster_rcnn_path,
            prefix='roi_head',
        ),
        type='mmdet.StandardRoIHead'),
    img_rpn_head=dict(
        anchor_generator=dict(
            # base_sizes will be set equal to strides if not specified differently
            ratios=[0.5, 1.0, 2.0, 3.0],
            scales=[4],
            strides=[4, 8, 16, 32, 64], # consistent with the FPN
            type='mmdet.AnchorGenerator'),
        bbox_coder=dict(
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            type='mmdet.DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='mmdet.L1Loss'),
        loss_cls=dict(loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=faster_rcnn_path,
            prefix='rpn_head',
        ),
        type='mmdet.RPNHead'),
    fusion_roi_head=dict(
        img_roi_layer=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='mmdet.SingleRoIExtractor'),
        lidar_bev_roi_layer=dict(
            featmap_strides=[2],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            grid_size=voxel_size[:2],
            point_cloud_range=point_cloud_range,
            type='AlignedBEVRoIExtractor'),
        bbox_head=dict(
            num_classes=3,
            in_channels=640,
            roi_feat_size=7,
            num_shared_fcs=2,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            # loss_cls=dict(
            #     type='mmdet.FocalLoss',
            #     use_sigmoid=True,
            #     gamma=2.0,
            #     alpha=0.25,
            #     loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                loss_weight=2.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            type='Residual3DBboxHead'),
        match_iou_threshold=0.5,
        train_cfg=dict(
            assigner=dict(  # TODO: multiple assigners for different classes?
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.5,
                type='mmdet.RandomSampler')),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50),
        type='LateFusionBEVRoIHead'),
    test_cfg=dict(
        pts_head=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.5, # not 0.01 only because is intermediate
            score_thr=0.05,
            min_bbox_size=0,
            nms_pre=1000,
            max_num=1000),
        img_rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000),
        img_rcnn=dict(
            max_per_img=1000,
            nms=dict(iou_threshold=0.7, type='nms'),
            score_thr=0.05)),
    train_cfg=dict(
        pts_head=dict(
            pos_weight=-1,
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)],
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.8, # only because is intermediate
            score_thr=0.05,
            min_bbox_size=0,
            nms_pre=1000,
            max_num=1000),
        img_rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='mmdet.RandomSampler'),
            max_per_img=1000,
            nms=dict(iou_threshold=0.9, type='nms'),
            score_thr=0.05),
        img_rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='mmdet.RandomSampler'),
            ),
        img_rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)))


##########################################################################
##########################   TRAINING SETTINGS   #########################
##########################################################################

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook', save_best="Kitti metric/pred_instances_3d/KITTI/Overall_3D_AP40_hard", rule="greater"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

custom_hooks = [
    dict(
        type='LayerWeightTransferHook',
        checkpoint_path=pointpillars_path,
        # voxel_encoder does not have the init_cfg parameter,
        # backone don't know why but is not initialized with the pretrained one
        layer_pairs=[('voxel_encoder', 'pts_voxel_encoder'), ('backbone', 'pts_backbone')]),
    dict(
        type='FreezeLayersBeforeTrainHook', 
        layer_names=['pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone', 'pts_neck', 
                     'pts_bbox_head', 'img_backbone', 'img_neck', 'img_rpn_head', 'img_roi_head'])]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False


##########################################################################
##########################   DATASET SETTINGS   ##########################
##########################################################################
dataset_type = 'KittiDataset'
data_root = '/mnt/proj2/dd-24-8/kitti_mmdet3d'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[1., 1.],
        translation_std=[0, 0, 0]),
    dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[1., 1.],
        translation_std=[0, 0, 0]),
    dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]
modality = dict(use_lidar=True, use_camera=True)
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            modality=modality,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
                # pts='training/velodyne', img='training/image_2'),
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
            # pts='training/velodyne', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        modality=modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
val_evaluator = dict(
    type='KittiMetric', ann_file='/mnt/proj2/dd-24-8/kitti_mmdet3d/kitti_infos_val.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
