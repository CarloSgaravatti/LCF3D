import argparse
import numpy as np
from pathlib import Path
import pickle
import time
import torch
from utils import read_kitti_calibration_data, calibration_to_torch

from stereo_inference import LateFusionStereoViewInferencer


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Expert late fusion demo")
    parser.add_argument("-output_dir", default=None, required=True, 
                        type=str, help="The directory where the outputs will be saved")
    parser.add_argument("-img_path_left", default=None, type=str, required=True,
                        help="File path for the inference left image")
    parser.add_argument("-img_path_right", default=None, type=str, required=True,
                        help="File path for the inference right image")
    parser.add_argument("-lidar_path", default=None, type=str, required=True,
                        help="File path for the inference point cloud")
    parser.add_argument("-calib_path", default=None, type=str, required=True,
                        help="File path for the calibration data (kitti format)")
    parser.add_argument("-late_fusion_config", default=None, type=str, required=True,
                        help="Late Fusion config path")
    parser.add_argument("-device", default='cpu', type=str, required=False,
                        help="Device for torch")
    parser.add_argument("--visualize", default=False, action='store_true',
                        help="Whether to produce and save the visualization files")
    
    args = parser.parse_args()
    inferencer = LateFusionStereoViewInferencer(
        late_fusion_cfg=args.late_fusion_config,
        device=args.device
    )
    
    calibration_data = read_kitti_calibration_data(args.calib_path)
    calibration_data = calibration_to_torch(calibration_data, device='cuda:0')
    lidar_to_cam = calibration_data['R0_rect'] @ calibration_data['Tr_velo_to_cam']
    cam_to_img_left = calibration_data['P2']
    cam_to_img_right = calibration_data['P3']
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        output = inferencer.visualize_predict(args.img_path_left, args.img_path_right, args.lidar_path, output_dir,
                                              lidar_to_cam, cam_to_img_left, cam_to_img_right)
        print(f'Visualizations saved at {str(output_dir)}')
    else:
        output = inferencer.predict(args.img_path_left, args.img_path_right, args.lidar_path,
                                    lidar_to_cam, cam_to_img_left, cam_to_img_right)
        
    with open(output_dir / 'predictions.pkl', 'wb') as pkl_file:
        pickle.dump(output, pkl_file)