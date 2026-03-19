auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
data_root_nuim = '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/'
data_root_nusc = '/DATA/nuScenes/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint=
            '/mnt/experiments/carlos00/models/kitti/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        init_cfg=dict(
            checkpoint=
            '/mnt/experiments/carlos00/models/kitti/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth',
            prefix='neck',
            type='Pretrained'),
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=10,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        init_cfg=dict(
            checkpoint=
            '/mnt/experiments/carlos00/models/kitti/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth',
            prefix='rpn_head',
            type='Pretrained'),
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
num_epochs = 12
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=5000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            5,
            9,
        ],
        type='MultiStepLR'),
]
pretrained_coco_path = '/mnt/experiments/carlos00/models/kitti/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/mnt/datasets_1/carlos00/nuscenes/nuinsseg_v1.0-trainval_val.json',
        backend_args=None,
        data_prefix=dict(img='/DATA/nuScenes/'),
        data_root='/DATA/nuScenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1600,
                900,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/mnt/datasets_1/carlos00/nuscenes/nuinsseg_v1.0-trainval_val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1600,
        900,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=12,
    dataset=dict(
        datasets=[
            dict(
                ann_file=
                '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/nuimages_v1.0-train.json',
                backend_args=None,
                data_prefix=dict(
                    img=
                    '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/'
                ),
                data_root=
                '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                metainfo=dict(classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ]),
                pipeline=[
                    dict(
                        backend_args=None,
                        imdecode_backend='pillow',
                        type='LoadImageFromFile'),
                    dict(
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=False),
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                1280,
                                720,
                            ),
                            (
                                1920,
                                1080,
                            ),
                        ],
                        type='RandomChoiceResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PackDetInputs'),
                ],
                type='CocoDataset'),
            dict(
                ann_file=
                '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/nuimages_v1.0-val.json',
                backend_args=None,
                data_prefix=dict(
                    img=
                    '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/'
                ),
                data_root=
                '/mnt/datasets_1/carlos00/nuimages/nuimages-v1.0-all-samples/',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                metainfo=dict(classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ]),
                pipeline=[
                    dict(
                        backend_args=None,
                        imdecode_backend='pillow',
                        type='LoadImageFromFile'),
                    dict(
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=False),
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                1280,
                                720,
                            ),
                            (
                                1920,
                                1080,
                            ),
                        ],
                        type='RandomChoiceResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PackDetInputs'),
                ],
                type='CocoDataset'),
            dict(
                ann_file=
                '/mnt/datasets_1/carlos00/nuscenes/nuinsseg_v1.0-trainval_train.json',
                backend_args=None,
                data_prefix=dict(img='/DATA/nuScenes/'),
                data_root='/DATA/nuScenes/',
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                metainfo=dict(classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ]),
                pipeline=[
                    dict(
                        backend_args=None,
                        imdecode_backend='pillow',
                        type='LoadImageFromFile'),
                    dict(
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=False),
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                1280,
                                720,
                            ),
                            (
                                1920,
                                1080,
                            ),
                        ],
                        type='RandomChoiceResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(type='PackDetInputs'),
                ],
                type='CocoDataset'),
        ],
        type='ConcatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        keep_ratio=True,
        scales=[
            (
                1280,
                720,
            ),
            (
                1920,
                1080,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/mnt/datasets_1/carlos00/nuscenes/nuinsseg_v1.0-trainval_val.json',
        backend_args=None,
        data_prefix=dict(img='/DATA/nuScenes/'),
        data_root='/DATA/nuScenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1600,
                900,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/mnt/datasets_1/carlos00/nuscenes/nuinsseg_v1.0-trainval_val.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/mnt/experiments/carlos00/mmdetection_work_dir/faster_rcnn_nuimages_nuscenes'
