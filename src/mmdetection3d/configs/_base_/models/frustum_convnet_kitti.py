voxel_size = [0.05, 0.05, 0.05]
pc_range = [-0, -40, -3, 80, 40, 1]
frustum_heights = [0.2, 0.4, 0.8, 1.6, 3.2]
frustum_features = [[32, 64, 128], [32, 64, 128], [64, 128, 256], [64, 128, 256], [128, 256, 512]]

model = dict(
    type='FrustumConvNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False),
    voxelization_layers=[
        # each "voxelization" layer encodes the coordinates of the points
        # in the frustum sections, the max number of points is not
        # specified since the voxelization is dynamic and more importantly
        # these are not really voxels, but it is implemented in this way
        # for convenience (each frustum section is treated as a very big
        # voxel to make FrustumConvNet compatible with mmdetection3d
        # structure). This completely respects the architecture, except
        # for the fact that frustum sections are not overlapping.
        dict(
            max_num_points=-1,
            point_cloud_range=pc_range,
            voxel_size=[0.1, 80, 4],
            max_voxels=(-1, -1)),
        dict(
            max_num_points=-1,
            point_cloud_range=pc_range,
            voxel_size=[0.2, 80, 4],
            max_voxels=(-1, -1)),
        dict(
            max_num_points=-1,  # max_points_per_voxel
            point_cloud_range=pc_range,
            voxel_size=[0.4, 80, 4],
            max_voxels=(-1, -1)),
        dict(
            max_num_points=-1,  # max_points_per_voxel
            point_cloud_range=pc_range,
            voxel_size=[0.8, 80, 4],
            max_voxels=(-1, -1)),
        dict(
            max_num_points=-1,  # max_points_per_voxel
            point_cloud_range=pc_range,
            voxel_size=[1.6, 80, 4],
            max_voxels=(-1, -1))],
        # dict(
        #     max_num_points=-1,  # max_points_per_voxel
        #     point_cloud_range=pc_range,
        #     voxel_size=[3.2, 80, 4],
        #     max_voxels=(-1, -1))],
    frustum_section_encoders=[
        dict(
            type='DynamicVFE',
            in_channels=4,
            feat_channels=[32, 64, 64],
            voxel_size=[0.1, 80, 4], # only cut over the depth of the frustum
            point_cloud_range=pc_range,
            mode='max',
            return_point_feats=False,),
        dict(
            type='DynamicVFE',
            in_channels=4,
            feat_channels=[32, 64, 128],
            voxel_size=[0.2, 80, 4], # only cut over the depth of the frustum
            point_cloud_range=pc_range,
            mode='max',
            return_point_feats=False,),
        dict(
            type='DynamicVFE',
            in_channels=4,
            feat_channels=[32, 64, 128],
            voxel_size=[0.4, 80, 4], # only cut over the depth of the frustum
            point_cloud_range=pc_range,
            mode='max',
            return_point_feats=False,),
        dict(
            type='DynamicVFE',
            in_channels=4,
            feat_channels=[64, 128, 256],
            voxel_size=[0.8, 80, 4], # only cut over the depth of the frustum
            point_cloud_range=pc_range,
            mode='max',
            return_point_feats=False,),
        dict(
            type='DynamicVFE',
            in_channels=4,
            feat_channels=[64, 128, 256],
            voxel_size=[1.6, 80, 4], # only cut over the depth of the frustum
            point_cloud_range=pc_range,
            mode='max',
            return_point_feats=False,)],
        # dict(
        #     type='DynamicVFE',
        #     in_channels=4,
        #     feat_channels=[128, 256, 512],
        #     voxel_size=[3.2, 80, 4], # only cut over the depth of the frustum
        #     point_cloud_range=pc_range,
        #     mode='max',
        #     return_point_feats=False,)],
    middle_encoders=[
        dict(
            type='PointPillarsScatter', 
            in_channels=64, 
            output_shape=[800, 1]),
        dict(
            type='PointPillarsScatter', 
            in_channels=128, 
            output_shape=[400, 1]),
        dict(
            type='PointPillarsScatter', 
            in_channels=128, 
            output_shape=[200, 1]),
        dict(
            type='PointPillarsScatter', 
            in_channels=256, 
            output_shape=[100, 1]),
        dict(
            type='PointPillarsScatter', 
            in_channels=256, 
            output_shape=[50, 1])],
    centroid_voxelization_layer=dict(
        max_num_points=1000,  # max_points_per_voxel
        point_cloud_range=pc_range,
        voxel_size=[0.4, 80, 4],
        max_voxels=(1000, 1000)),
    centroid_section_encoder=dict(type='HardSimpleVFE'),
    centroid_middle_encoder=dict(
        type='PointPillarsScatter', 
        in_channels=3, 
        output_shape=[200, 1]),
    backbone=dict(
        type='FrustumConvNetFCN',
        in_channels=[64, 128, 128, 256, 256], #, 512],
        out_channels_conv=[64, 128, 128, 256, 256], #, 512],
        conv_strides=[2, 2, 2, 2, 2],
        deconv_num=4,
        deconv_channels=[128, 128, 256, 256], #, 512],
        deconv_strides=[1, 2, 4, 8],
        deconv_kernel_sizes=[1, 2, 4, 8]),
    bbox_head=dict(
        type='FrustumConvNetBboxHead',
        point_cloud_range=pc_range,
        shared_conv_channels=[768],
        in_channels=768,
        num_classes=3,
        reg_decoded_bbox=False,
        anchor_generator=dict(
            type='FrustumAnchorGenerator',
            frustum_range=pc_range,
            heights=[-0.6, -0.6, -1.78],
            anchor_sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57]),
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
            loss_weight=0.2)),
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=True),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=10))
