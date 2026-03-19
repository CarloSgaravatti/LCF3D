_base_ = [
    '../_base_/models/frustum_convnet_kitti.py',
    '../_base_/datasets/kitti-crop-aligned-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

model_wrapper_cfg = dict(type='MMDistributedDataParallel')

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')