#!/usr/bin/bash
#SBATCH --job-name LCF3D-NuScenes-Eval
#SBATCH --account EU-25-52
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 4:00:00
#SBATCH --error myJob_nuscenes_eval.err
#SBATCH --output myJob_nuscenes_eval.out
#SBATCH --gres=gpu:1
#SBATCH --nodes 1

module purge
source /mnt/proj2/eu-25-52/conda_envs/etc/profile.d/conda.sh
conda activate mmdet3d-cu117

echo "PYTHON: $(which python)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Update these paths for your setup
NUSCENES_ROOT=/DATA/nuScenes
OUT_DIR=$HOME/experiments/lcf3d_nuscenes_eval
CFG=$HOME/LCF3D/src/example_cfg.json

python $HOME/LCF3D/src/eval_nuscenes.py \
  -output_dir $OUT_DIR \
  -nuscenes_root_path $NUSCENES_ROOT \
  -late_fusion_config $CFG \
  -device cuda:0 \
  --version v1.0-trainval \
  --eval_set val \
  --eval_version detection_cvpr_2019 \
  --sweep_num 10 \
  --cameras CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT,CAM_BACK_LEFT,CAM_BACK,CAM_BACK_RIGHT
