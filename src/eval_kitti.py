import sys
sys.path.append("/home/it4i-carlos00/3d_object_detection")

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import read_kitti_calibration_data, calibration_to_torch
from stereo_inference import LateFusionStereoViewInferencer
from single_view_inference import LateFusionSingleViewInferencer

from mmdet3d.evaluation.metrics import KittiNuScenesMetric, KittiMetric

try:
    from pyquaternion import Quaternion
except Exception:  # pragma: no cover
    Quaternion = None


LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def _load_kitti_points(
    velo_pc: str,
    *,
    dtype: np.dtype,
    num_features: int,
    points_dim_mode: str,
    intensity_divisor: float = None,
    intensity_scale: float = 1.0,
    time_value: float = 0.0,
) -> np.ndarray:
    """Load a KITTI-format .bin and normalize it into a NxD array.

    points_dim_mode:
      - 'xyzi': output Nx4 (x,y,z,intensity)
      - 'xyzit': output Nx5 (x,y,z,intensity,time)
      - 'xyzt': output Nx4 (x,y,z,time)
    """
    pts_raw = np.fromfile(velo_pc, dtype=dtype).reshape((-1, num_features))

    if points_dim_mode == 'xyzi':
        pts = pts_raw[:, :4].copy()
        if intensity_divisor is not None:
            pts[:, 3] /= float(intensity_divisor)
        pts[:, 3] *= float(intensity_scale)
        return pts.astype(np.float32)

    if points_dim_mode == 'xyzit':
        if pts_raw.shape[1] >= 5:
            pts = pts_raw[:, :5].copy()
        else:
            pts = pts_raw[:, :4].copy()
            pts = np.insert(pts, 4, float(time_value), axis=1)
        if intensity_divisor is not None:
            pts[:, 3] /= float(intensity_divisor)
        pts[:, 3] *= float(intensity_scale)
        pts[:, 4] = float(time_value) if pts_raw.shape[1] < 5 else pts[:, 4]
        return pts.astype(np.float32)

    if points_dim_mode == 'xyzt':
        # Prefer using an explicit time column if present (5th feature). Otherwise inject a constant time.
        if pts_raw.shape[1] >= 5:
            pts = np.stack([pts_raw[:, 0], pts_raw[:, 1], pts_raw[:, 2], pts_raw[:, 4]], axis=1)
        else:
            pts = pts_raw[:, :4].copy()
            pts[:, 3] = float(time_value)
        return pts.astype(np.float32)

    raise ValueError(f"Unknown --kitti_points_dim_mode '{points_dim_mode}'")


def _run_kitti_eval(args) -> None:
    kitti_root_path = Path(args.kitti_root_path)
    left_path = kitti_root_path / 'image_2'
    right_path = kitti_root_path / 'image_3'
    velo_path = kitti_root_path / args.kitti_velodyne_folder
    calib_path = kitti_root_path / 'calib'

    output_dir = Path(args.output_dir) / time.strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'data'
    metrics_path = output_dir / 'metrics.json'

    predictions_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(output_dir / 'log.txt'),
        filemode='w',
        format=LOGGING_FORMATTER,
        level=logging.INFO,
        force=True,
    )

    if args.view == 'stereo':
        inferencer = LateFusionStereoViewInferencer(
            late_fusion_cfg=args.late_fusion_config,
            device=args.device,
        )
    else:
        inferencer = LateFusionSingleViewInferencer(
            late_fusion_cfg=args.late_fusion_config,
            device=args.device,
        )

    with open(args.validation_split_path, 'r') as split_file:
        validation_ids = [val_id.rstrip('\n') for val_id in split_file.readlines()]
        
    with open(args.late_fusion_config, 'r') as cfg_file:
        late_fusion_cfg = json.load(cfg_file)
    
    late_fusion_cfg['args'] = vars(args)
    with open(output_dir / 'config.json', 'w') as out_cfg_file:
        json.dump(late_fusion_cfg, out_cfg_file, indent=4)

    results_list = []
    initial_time = time.time()

    kitti_to_nu_lidar = None
    kitti_to_nu_lidar_inv = None
    if args.kitti_to_nuscenes_lidar:
        if Quaternion is None:
            raise ImportError('pyquaternion is required for --kitti_to_nuscenes_lidar')
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

    dtype = np.float64 if args.kitti_is_float64_bin else np.float32
    num_features = args.kitti_num_features

    for i, val_id in enumerate(tqdm(validation_ids)):
        if i % 100 == 0:
            logging.info(
                f'Progress: {100 * i / max(1, len(validation_ids)):.2f}%, '
                f'elapsed {int(time.time() - initial_time)}s'
            )

        left_image = str(left_path / f'{val_id}.png')
        right_image = str(right_path / f'{val_id}.png')
        velo_pc = str(velo_path / f'{val_id}.bin')

        calibration_data = read_kitti_calibration_data(calib_path / f'{val_id}.txt')
        calibration_data = calibration_to_torch(calibration_data, device=args.device)

        # Optional: override points instead of letting the pipeline load from file.
        # This is needed for (a) KITTI-format datasets with non-standard feature dims, and
        # (b) running nuScenes-trained models on KITTI-format data (rotate KITTI lidar -> nuScenes lidar).
        need_preload_points = (
            args.preload_points
            or args.kitti_to_nuscenes_lidar
            or args.kitti_is_float64_bin
            or args.kitti_num_features != 4
            or args.kitti_points_dim_mode != 'xyzi'
            or args.kitti_intensity_divisor is not None
            or args.kitti_intensity_scale != 1.0
        )

        points = None
        if need_preload_points:
            pts = _load_kitti_points(
                velo_pc,
                dtype=dtype,
                num_features=num_features,
                points_dim_mode=args.kitti_points_dim_mode,
                intensity_divisor=args.kitti_intensity_divisor,
                intensity_scale=args.kitti_intensity_scale,
                time_value=args.kitti_time_value,
            )

            if args.kitti_to_nuscenes_lidar:
                # Rotate KITTI lidar -> nuScenes lidar
                pts[:, :3] = pts[:, :3] @ kitti_to_nu_lidar.rotation_matrix.T

                # Adjust lidar->cam so projections remain consistent.
                calibration_data['Tr_velo_to_cam'][:3, :3] = (
                    calibration_data['Tr_velo_to_cam'][:3, :3] @
                    torch.tensor(kitti_to_nu_lidar_inv.rotation_matrix, dtype=torch.float32, device=args.device)
                )

            points = pts

        lidar_to_cam = calibration_data['R0_rect'] @ calibration_data['Tr_velo_to_cam']
        cam_to_img_left = calibration_data['P2']
        cam_to_img_right = calibration_data['P3']

        if args.view == 'stereo':
            detection_out = inferencer.predict(
                left_image,
                right_image,
                velo_pc,
                lidar_to_cam,
                cam_to_img_left,
                cam_to_img_right,
                points=points,
            )
        else:
            detection_out = inferencer.predict(
                left_image,
                velo_pc,
                lidar_to_cam,
                cam_to_img_left,
                points=points,
            )

        bboxes_3d = detection_out.bboxes_3d.to('cpu')
        scores_3d = detection_out.scores_3d.to('cpu')
        labels_3d = detection_out.labels_3d.to('cpu')

        if args.kitti_to_nuscenes_lidar and bboxes_3d.tensor.shape[0] > 0:
            # Rotate predictions back to KITTI lidar for KITTI metric.
            bboxes_3d.rotate(kitti_to_nu_lidar_inv.rotation_matrix.T)

        result = {
            'pred_instances_3d': {
                'bboxes_3d': bboxes_3d,
                'scores_3d': scores_3d,
                'labels_3d': labels_3d,
            },
            'sample_idx': i,
        }
        results_list.append(result)

    logging.info('Starting evaluation')

    results = {}
    kitti_metric = KittiMetric(
        ann_file=args.annotation_file_eval,
        metric=inferencer.detector3d.cfg.val_evaluator.get('metric', 'bbox'),
        backend_args=inferencer.detector3d.cfg.val_evaluator.get('backend_args', None),
        submission_prefix=str(predictions_path) if args.save_predictions else None,
        format_only=args.test,
    )

    # Use KITTI meta by default (helps cross-dataset runs where detector3d.dataset_meta isn't KITTI-like).
    if args.kitti_classes is not None:
        classes = [c.strip() for c in args.kitti_classes.split(',') if c.strip()]
        kitti_metric._dataset_meta = {'classes': classes}
    else:
        kitti_metric._dataset_meta = getattr(
            inferencer.detector3d,
            'dataset_meta',
            {'classes': ['Pedestrian', 'Cyclist', 'Car']},
        )

    metrics_dict = kitti_metric.compute_metrics(results_list)
    results['KITTI_Metrics'] = metrics_dict
    
    nusc_metric = KittiNuScenesMetric(
        ann_file=args.annotation_file_eval,
        metric='bbox',
        backend_args=None
    )
    nusc_metric._dataset_meta = kitti_metric._dataset_meta
    metrics_dict = nusc_metric.compute_metrics(results_list)
    
    results['nuScenes_Metrics'] = metrics_dict

    with open(metrics_path, 'w') as metrics_file:
        json.dump(results, metrics_file, indent=4)

    logging.info(f'Results: {results}')

    with open(output_dir / 'predictions.pkl', 'wb') as fp:
        pickle.dump(results_list, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KITTI-format evaluation entrypoint for LCF3D (supports cross-dataset via input adjustments).'
    )

    parser.add_argument('-output_dir', default='/mnt/it4i-carlos00/experiments/expert_late_fusion_predictions/data',
                        type=str, help='Directory where outputs will be saved')
    parser.add_argument('-kitti_root_path', required=True, type=str,
                        help='KITTI-format dataset root path (can also be a converted NuScenes-to-KITTI dataset)')
    parser.add_argument('-late_fusion_config', required=True, type=str,
                        help='Late fusion config path')
    parser.add_argument('-device', default='cuda:0', type=str,
                        help='Device (e.g. cpu, cuda:0)')

    parser.add_argument('-validation_split_path',
                        default='/home/it4i-carlos00/3d_object_detection/src/expert_late_fusion/val.txt',
                        type=str,
                        help='KITTI split file with sample ids')
    parser.add_argument('-annotation_file_eval',
                        default='/mnt/proj2/dd-24-8/kitti_mmdet3d/kitti_infos_val.pkl',
                        type=str,
                        help='KITTI infos pkl for KittiMetric')
    parser.add_argument('--save_predictions', default=False, action='store_true',
                        help='Write KITTI-format predictions to output_dir/data')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Format-only (no metric computation) where supported')
    parser.add_argument('--view', default='stereo', choices=['stereo', 'single'],
                        help='KITTI inference mode')
    parser.add_argument('--kitti_velodyne_folder', default='velodyne_reduced', type=str,
                        help='Velodyne folder name under KITTI root (e.g. velodyne_reduced)')
    parser.add_argument('--kitti_classes', default=None, type=str,
                        help='Comma-separated KITTI class names (overrides dataset meta classes)')

    # Cross-dataset helper (KITTI lidar -> nuScenes lidar convention)
    parser.add_argument('--kitti_to_nuscenes_lidar', default=False, action='store_true',
                        help='Rotate KITTI points/calib into nuScenes lidar frame before inference, then rotate boxes back for KITTI eval')

    # Point loading / formatting (useful when KITTI-format bins have non-standard feature dims)
    parser.add_argument('--preload_points', default=False, action='store_true',
                        help='Always load points with numpy and pass `points=` to the model instead of relying on pipeline file loading')
    parser.add_argument('--kitti_is_float64_bin', action='store_true', default=False,
                        help='Load KITTI .bin as float64 (rare)')
    parser.add_argument('--kitti_num_features', default=4, type=int,
                        help='Number of features per KITTI point (4 or 5)')
    parser.add_argument('--kitti_points_dim_mode', default='xyzi', choices=['xyzi', 'xyzit', 'xyzt'],
                        help='How to format KITTI points before passing to the model')
    parser.add_argument('--kitti_intensity_divisor', default=None, type=float,
                        help='If set, divides intensity by this value after loading (e.g. 255.0)')
    parser.add_argument('--kitti_intensity_scale', default=1.0, type=float,
                        help='Multiplies intensity by this value after optional division (e.g. 255.0)')
    parser.add_argument('--kitti_time_value', default=0.0, type=float,
                        help='Constant time value to inject when building xyzt/xyzit without an explicit time column')

    args = parser.parse_args()

    _run_kitti_eval(args)
