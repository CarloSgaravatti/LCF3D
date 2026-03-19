_base_ = [
    '../_base_/datasets/kitti-crop-aligned-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FrustumLocalizationModel',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False),
    in_channels=4,
    num_classes=3,
    point_net_channels=[(64, 64), (64, 128, 1024)],
    final_channels=[512, 256],
    loss_cls=dict(
        type='mmdet.CrossEntropyLoss', use_sigmoid=False,
        loss_weight=1.0),
    loss_bbox=dict(
        type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0))

lr = 0.0001
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=10, norm_type=2))
