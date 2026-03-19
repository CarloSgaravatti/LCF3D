_base_ = [
    '../_base_/models/pointpillars_hv_fpn_nus.py',
    '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
train_cfg = dict(val_interval=24)

voxel_size = [0.25, 0.25, 8]
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            deterministic=False,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.3,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=200,
        max_num=100))
