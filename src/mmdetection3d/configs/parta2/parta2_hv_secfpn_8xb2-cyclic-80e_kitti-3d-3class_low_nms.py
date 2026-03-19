_base_ = [
    './parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class.py'
]

model=dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.2,
            score_thr=0.1)))