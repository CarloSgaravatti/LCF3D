import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes, LidarPointCloud
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes

from multi_view_inference import LateFusionMultiViewInferencer
from mmdet3d.evaluation.metrics.nuscenes_metric import (
    output_to_nusc_box,
    lidar_nusc_box_to_global,
    NuScenesMetric,
)

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def _to_4x4(rotation: List[float], translation: List[float]) -> np.ndarray:
    tr = np.eye(4, dtype=np.float64)
    tr[:3, :3] = Quaternion(rotation).rotation_matrix
    tr[:3, 3] = np.array(translation, dtype=np.float64)
    return tr


def _to_3x4_intrinsic(camera_intrinsic: List[List[float]]) -> np.ndarray:
    k = np.array(camera_intrinsic, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"camera_intrinsic must be 3x3, got {k.shape}")
    return np.insert(k, 3, 0.0, axis=1)


def _collect_sample_tokens(nusc: NuScenes, scene_names: List[str]) -> List[str]:
    sample_tokens: List[str] = []
    for scene_name in scene_names:
        scene = next(sc for sc in nusc.scene if sc['name'] == scene_name)
        token = scene['first_sample_token']
        while token != '':
            sample_tokens.append(token)
            sample = nusc.get('sample', token)
            token = sample['next']
    return sample_tokens


def _get_nuscenes_attr(box, class_names: List[str]) -> Tuple[str, str]:
    name = class_names[box.label]
    speed = float(np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2))

    if speed > 0.2:
        if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
            attr = 'vehicle.moving'
        elif name in ['bicycle', 'motorcycle']:
            attr = 'cycle.with_rider'
        else:
            attr = NuScenesMetric.DefaultAttribute[name]
    else:
        if name in ['pedestrian']:
            attr = 'pedestrian.standing'
        elif name in ['bus']:
            attr = 'vehicle.stopped'
        else:
            attr = NuScenesMetric.DefaultAttribute[name]

    return name, attr


def _load_multisweep_points(nusc: NuScenes, sample: dict, sweep_num: int, use_intensity: bool) -> np.ndarray:
    pcl, times = LidarPointCloud.from_file_multisweep(nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', sweep_num)
    points = pcl.points  # shape (4, N) [x,y,z,intensity]

    if not use_intensity:
        points[3, :] = times
        pts = points[:4, :].T  # (N,4) [x,y,z,time]
    else:
        pts = np.concatenate([points, times.reshape(1, -1)], axis=0)
        pts = pts[:5, :].T  # (N,5) [x,y,z,intensity,time]

    return pts.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NuScenes multi-view evaluation using official NuScenes API + NuScenesEval (LCF3D inference).'
    )
    parser.add_argument('-output_dir', required=True, type=str,
                        help='Directory where outputs (results json + metrics) will be saved')
    parser.add_argument('-nuscenes_root_path', required=True, type=str,
                        help='NuScenes dataroot (e.g. /DATA/nuScenes)')
    parser.add_argument('-late_fusion_config', required=True, type=str,
                        help='Late fusion config path (must be compatible with LateFusionMultiViewInferencer)')
    parser.add_argument('-device', default='cuda:0', type=str,
                        help='Device (e.g. cuda:0)')

    parser.add_argument('--version', default='v1.0-trainval', type=str,
                        help='NuScenes version (v1.0-trainval or v1.0-mini)')
    parser.add_argument('--eval_set', default='val', type=str,
                        help='Evaluation split name for NuScenesEval (val or mini_val)')
    parser.add_argument('--eval_version', default='detection_cvpr_2019', type=str,
                        help='NuScenes detection config (e.g. detection_cvpr_2019)')

    parser.add_argument('--cameras',
                        default='CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT,CAM_BACK_LEFT,CAM_BACK,CAM_BACK_RIGHT',
                        type=str, help='Comma-separated camera keys to use')
    parser.add_argument('--class_names',
                        default='car,truck,trailer,bus,construction_vehicle,bicycle,motorcycle,pedestrian,traffic_cone,barrier',
                        type=str,
                        help='Comma-separated class names (index must match model labels)')

    parser.add_argument('--sweep_num', default=10, type=int,
                        help='Number of sweeps to use (as in nuscenes_eval_frustum_mask.py)')
    parser.add_argument('--use_intensity', default=False, action='store_true',
                        help='If set, build points with intensity + time (Nx5). Otherwise Nx4 with time in 4th dim.')
    parser.add_argument('--max_samples', default=None, type=int,
                        help='Optional cap for debugging')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(output_dir / 'log.txt'),
        filemode='w',
        format=LOGGING_FORMATTER,
        level=logging.INFO,
        force=True,
    )

    cameras = [c.strip() for c in args.cameras.split(',') if c.strip()]
    class_names = [c.strip() for c in args.class_names.split(',') if c.strip()]

    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root_path, verbose=True)

    inferencer = LateFusionMultiViewInferencer(
        late_fusion_cfg=args.late_fusion_config,
        device=args.device,
    )

    if hasattr(inferencer, 'num_views') and inferencer.num_views != len(cameras):
        raise ValueError(
            f"Config num_views={inferencer.num_views} but --cameras specifies {len(cameras)} views. "
            "Update your late_fusion_config (num_views) or the --cameras list to match."
        )

    split_scenes = create_splits_scenes()
    if args.eval_set not in split_scenes:
        raise ValueError(f"Unknown eval_set '{args.eval_set}'. Available: {list(split_scenes.keys())}")

    scene_names = split_scenes[args.eval_set]
    sample_tokens = _collect_sample_tokens(nusc, scene_names)

    if args.max_samples is not None:
        sample_tokens = sample_tokens[:args.max_samples]

    results_dict = {
        'meta': {
            'use_camera': True,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        },
        'results': {},
    }

    detection_config = config_factory(args.eval_version)
    start_time = time.time()

    for idx, sample_token in enumerate(tqdm(sample_tokens)):
        if idx % 50 == 0:
            logging.info(
                f'Progress: {100 * idx / max(1, len(sample_tokens)):.2f}%, '
                f'elapsed {int(time.time() - start_time)}s'
            )

        sample = nusc.get('sample', sample_token)
        points = _load_multisweep_points(nusc, sample, args.sweep_num, args.use_intensity)

        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ego_pose_lidar = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        lidar_tr = _to_4x4(lidar_calib['rotation'], lidar_calib['translation'])
        ego_pose_tr = _to_4x4(ego_pose_lidar['rotation'], ego_pose_lidar['translation'])
        pc_total_tr = ego_pose_tr @ lidar_tr

        img_files: List[str] = []
        extrinsics: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []

        for cam_key in cameras:
            cam_data = nusc.get('sample_data', sample['data'][cam_key])
            ego_pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

            ego_pose_cam_tr = _to_4x4(ego_pose_cam['rotation'], ego_pose_cam['translation'])
            camera_tr = _to_4x4(cam_calib['rotation'], cam_calib['translation'])

            extrinsic_matrix = np.linalg.inv(camera_tr) @ np.linalg.inv(ego_pose_cam_tr) @ pc_total_tr
            intrinsic_matrix = _to_3x4_intrinsic(cam_calib['camera_intrinsic'])

            extrinsics.append(torch.tensor(extrinsic_matrix, dtype=torch.float32, device=args.device))
            intrinsics.append(torch.tensor(intrinsic_matrix, dtype=torch.float32, device=args.device))
            img_files.append(os.path.join(nusc.dataroot, cam_data['filename']))

        detection_out = inferencer.predict(
            img_files=img_files,
            pc_file=os.path.join(nusc.dataroot, lidar_data['filename']),
            lidar_to_cam=extrinsics,
            cam_to_img=intrinsics,
            points=points,
        )

        if detection_out.bboxes_3d.tensor.shape[0] == 0:
            boxes_global = []
        else:
            detection = {
                'bboxes_3d': detection_out.bboxes_3d.to('cpu'),
                'scores_3d': detection_out.scores_3d.to('cpu'),
                'labels_3d': detection_out.labels_3d.to('cpu'),
            }
            boxes, _ = output_to_nusc_box(detection)
            info = {
                'lidar_points': {'lidar2ego': lidar_tr},
                'ego2global': ego_pose_tr,
            }
            boxes_global = lidar_nusc_box_to_global(info, boxes, class_names, detection_config)

        serialized_boxes = []
        for box in boxes_global:
            # NuScenesEval rejects any box with a non-positive or NaN dimension
            # ("Error: sample_result sizes must be >0.") and aborts the whole eval.
            # A few recovered/frustum boxes can come out degenerate, so drop them here.
            wlh = box.wlh.tolist()
            center = box.center.tolist()
            if any((v is None) or np.isnan(v) for v in wlh + center) or any(v <= 0 for v in wlh):
                continue
            name, attr = _get_nuscenes_attr(box, class_names)
            serialized_boxes.append({
                'sample_token': sample_token,
                'translation': center,
                'size': wlh,
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': float(box.score),
                'attribute_name': attr,
            })

        results_dict['results'][sample_token] = serialized_boxes

    results_path = output_dir / 'results_nusc.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    nusc_eval = NuScenesEval(
        nusc,
        config=detection_config,
        result_path=str(results_path),
        eval_set=args.eval_set,
        output_dir=str(output_dir),
        verbose=False,
    )
    metrics = nusc_eval.main(render_curves=False)

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info('mAP: %.4f' % (metrics['mean_ap']))
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    for tp_name, tp_val in metrics['tp_errors'].items():
        logging.info('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    logging.info('NDS: %.4f' % (metrics['nd_score']))
    logging.info('Eval time: %.1fs' % metrics['eval_time'])

    logging.info('Per-class results:')
    logging.info('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
    class_aps = metrics['mean_dist_aps']
    class_tps = metrics['label_tp_errors']
    for class_name in class_aps.keys():
        logging.info(
            '%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f' % (
                class_name,
                class_aps[class_name],
                class_tps[class_name]['trans_err'],
                class_tps[class_name]['scale_err'],
                class_tps[class_name]['orient_err'],
                class_tps[class_name]['vel_err'],
                class_tps[class_name]['attr_err'],
            )
        )
