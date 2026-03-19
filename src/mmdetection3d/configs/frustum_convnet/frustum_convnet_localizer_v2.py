_base_ = [
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'FrustumDataset'
data_root = '/mnt/proj2/dd-24-8/frustum_datasets/v1/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None
point_cloud_range = [-0, -40, -3, 80, 40, 1]

model = dict(
    type='FrustumConvNetLocalizerV2',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    voxelization_layer=dict(
        max_num_points=-1,  # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=[0.4, 80, 4],
        max_voxels=(-1, -1)),
    frustum_section_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 128, 256],
        voxel_size=[0.4, 80, 4], # only cut over the depth of the frustum
        point_cloud_range=point_cloud_range,
        mode='max',
        return_point_feats=False,),
    middle_encoder=dict(
        type='PointPillarsScatter', 
        in_channels=256,
        output_shape=[200, 1]),
    bbox_head=dict(
        type='FrustumConvNetLocalizerBboxHeadV2',
        point_cloud_range=point_cloud_range,
        num_classes=3,
        anchor_sizes=[
            [0.8, 0.6, 1.73], 
            [1.76, 0.6, 1.73], 
            [3.9, 1.6, 1.56]],
        rotations=[0, 0.523, 1.046, 1.57, 2.093, 2.617],
        shared_conv_channels=[512, 512, 1024, 1024],
        kernel_sizes=[5, 3, 1, 5, 3, 1],
        strides=[2, 2, 1, 2, 1, 1],
        in_channels=256,
        cls_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        center_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        head_cls_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0),
        head_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0),
        size_cls_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        size_res_loss=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0),
        dir_loss=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2),
        gamma_corner=0.1,
        corner_loss=True))

lr = 0.0001
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=10, norm_type=2))

train_pipeline = [
    dict(
        type='LoadPointsFromDict',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.0, 0.0, 0.00],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.0, 0.0], # no rotation since dataset is in frustum coordinates
        scale_ratio_range=[1.0, 1.0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask'],
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape',
                   'scale_factor', 'flip', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                   'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                   'pcd_rotation_angle', 'lidar_path',
                   'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                   'cam2global', 'crop_offset', 'img_crop_offset',
                   'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                   'num_ref_frames', 'num_views', 'ego2global',
                   'axis_align_matrix', 'one_hot_vector'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromDict',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape',
                   'scale_factor', 'flip', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                   'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                   'pcd_rotation_angle', 'lidar_path',
                   'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                   'cam2global', 'crop_offset', 'img_crop_offset',
                   'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                   'num_ref_frames', 'num_views', 'ego2global',
                   'axis_align_matrix', 'one_hot_vector'))
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromDict',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    # need to load annotations to make the evaluation simpler to code
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    # dict(type='PointSample', num_points=256, replace=False),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask'],
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape',
                   'scale_factor', 'flip', 'pcd_horizontal_flip',
                   'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                   'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                   'pcd_rotation_angle', 'lidar_path',
                   'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                   'cam2global', 'crop_offset', 'img_crop_offset',
                   'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                   'num_ref_frames', 'num_views', 'ego2global',
                   'axis_align_matrix', 'one_hot_vector'))
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        # dataset=dict(type='CBGSDataset'),
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'kitti_frustum_info_train.pkl',
            pipeline=train_pipeline,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_frustum_info_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=metainfo,
        load_eval_anns=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'kitti_frustum_info_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='FrustumLocalizationMetric',
    box_type_3d='LiDAR',
    with_cls_out=False,
    classes=['Pedestrian', 'Cyclist', 'Car'],
    iou_thresholds=[0.7, 0.5, 0.5])
test_evaluator = val_evaluator

# optimizer
lr = 0.0005
epoch_num = 50
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.001, type='LinearLR'),
    dict(type='ReduceOnPlateauLR', factor=0.5, patience=5, rule='greater', monitor='IoU_3D', min_value=1e-5),
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)
val_cfg = dict()
test_cfg = dict()

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook', save_best="IoU_3D", rule="greater"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))