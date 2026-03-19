_base_ = [
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

pcd_range = [0, -3.84, -3, 76.8, 3.84, 1]
# dataset settings
dataset_type = 'KittiDataset'
data_root = '/mnt/proj2/dd-24-8/kitti_crops_aligned/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None

voxel_size = [0.16, 0.16, 4]

model = dict(
    type='FrustumPillarLocalizer',
    dropout_proba=0.2,
    num_channels_head=512,
    num_classes=3,
    anchor_by_class=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
    height_by_class=[-0.6, -0.6, -1.78],
    cls_weights=[2, 4, 0.5],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=pcd_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=pcd_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[480, 48]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5, 5],
        layer_strides=[2, 2, 2, 2],
        out_channels=[64, 128, 256, 512]),
    loss_cls=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        loss_weight=0.5),
    loss_bbox=dict(
        type='mmdet.SmoothL1Loss',
        beta=1.0 / 9.0,
        loss_weight=2.0),
    loss_dir=dict(
        type='mmdet.CrossEntropyLoss', loss_weight=0.2),)

lr = 0.0001
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=10, norm_type=2))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[0.1, 0.1, 0.01],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.0, 0.0], # no rotation since dataset is in frustum coordinates
        scale_ratio_range=[1.0, 1.0]), #[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=pcd_range),
    dict(type='ObjectRangeFilter', point_cloud_range=pcd_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
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
                type='PointsRangeFilter', point_cloud_range=pcd_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
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
                type='PointsRangeFilter', point_cloud_range=pcd_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        # dataset=dict(type='CBGSDataset'),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
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
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
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
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne'),
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='FrustumLocalizationMetric',
    box_type_3d='LiDAR',
    classes=['Pedestrian', 'Cyclist', 'Car'],
    iou_thresholds=[0.7, 0.5, 0.5])
test_evaluator = val_evaluator

# optimizer
lr = 0.0001
epoch_num = 100
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=10, norm_type=2))
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.001, type='LinearLR'),
    dict(type='ReduceOnPlateauLR', factor=0.5, patience=5, rule='greater', monitor='IoU_3D', min_value=5e-6),
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

