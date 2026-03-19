_base_ = [
    '../_base_/models/pointpillars_hv_fpn_nus.py',
    '../_base_/datasets/nus-3d-reduced-class.py', 
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
train_cfg = dict(val_interval=24)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False)

model = dict(
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.3,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=200,
        max_num=100))
