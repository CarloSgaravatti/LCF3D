auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/mnt/proj2/dd-24-8'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', save_best="coco/bbox_mAP", rule="greater"),
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
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet101', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4, # resnet stages (initial conv1 is not considered a stage)
        out_indices=(0, 1, 2, 3,),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_size_divisor=32,
        std=[58.395, 57.12, 57.375],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[256, 512, 1024, 2048],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=3,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            # base_sizes will be set equal to strides if not specified differently
            ratios=[0.5, 1.0, 2.0, 3.0],
            scales=[4],
            strides=[4, 8, 16, 32, 64], # consistent with the FPN
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.7),
            max_per_img=100),
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
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(type='ReduceOnPlateauLR', factor=0.1, patience=10, rule='greater', monitor='coco/bbox_mAP', min_value=1e-5),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/mnt/proj2/dd-24-8/val_annot_coco.json',
        backend_args=None,
        data_prefix=dict(img='data_object_image_2/training/image_2'),
        metainfo=dict(classes=('Pedestrian', 'Cyclist', 'Car')),
        data_root='/mnt/proj2/dd-24-8',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs'), #meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/mnt/proj2/dd-24-8/val_annot_coco.json',
    backend_args=None,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'), #meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
]
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=1)
albu_transforms = [
    # dict(type='HorizontalFlip', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='Equalize', p=0.5)
        ],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomRain', slant_lower=-20, slant_upper=20, drop_length=10, 
                 brightness_coefficient=0.9, blur_value=3, p=0.15),
            dict(type='RandomShadow', p=0.25),
            dict(type='RandomSunFlare', src_radius=300, num_flare_circles_upper=20, flare_roi=(0.3, 0, 0.7, 0.3)),
            dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.3, p=0.5),
        ],
        p=0.3),
    dict(type='GaussNoise', var_limit=(2, 10), p=0.5),
    dict(type='MotionBlur', blur_limit=9, p=0.1),
]
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file='/mnt/proj2/dd-24-8/train_annot_coco.json',
        backend_args=None,
        data_prefix=dict(img='data_object_image_2/training/image_2'),
        metainfo=dict(classes=('Pedestrian', 'Cyclist', 'Car')),
        data_root='/mnt/proj2/dd-24-8',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(keep_ratio=True, scale=[(1333, 640), (1333, 800)], type='RandomResize'),
            dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
            # dict(prob=0.5, type='RandomFlip'),
            dict(
                type='Albu',
                transforms=albu_transforms,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap={
                    'img': 'image',
                    'gt_bboxes': 'bboxes',
                },
                skip_img_without_anno=True),
            dict(type='PackDetInputs') #, meta_keys=('img', 'gt_bboxes', 'gt_labels')),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(keep_ratio=True, scale=[(1333, 640), (1333, 800)], type='RandomResize'),
    dict(keep_ratio=True, scale=(1333, 800,), type='Resize'),
    # dict(prob=0.5, type='RandomFlip'),
    dict(
        type='Albu',
        transforms=albu_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            #'gt_labels': 'labels',
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs') #, meta_keys=('image', 'bboxes', 'labels')),
]
val_cfg = dict(type='ValLoop')
val_dataloader = test_dataloader
val_evaluator = test_evaluator
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
