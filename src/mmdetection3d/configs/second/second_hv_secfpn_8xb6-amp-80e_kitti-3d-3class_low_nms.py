_base_ = [
    './second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
]

model=dict(
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.3,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=100))