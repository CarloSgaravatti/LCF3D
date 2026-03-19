#!/usr/bin/bash
#SBATCH --job-name ExpertLateFusion
#SBATCH --account EU-25-52
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 1:00:00
#SBATCH --error myJob_pp.err
#SBATCH --output myJob_pp.out
#SBATCH --gres=gpu:1
#SBATCH --nodes 1

module purge
source /mnt/proj2/eu-25-52/conda_envs/etc/profile.d/conda.sh
conda activate mmdet3d-cu117

echo "PYTHON: $(which python)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

python $HOME/LCF3D/src/eval_kitti.py \
	-output_dir $HOME/experiments/code_to_publish_single_view \
	-kitti_root_path /mnt/proj3/eu-25-19/KITTI/training \
	-annotation_file_eval /mnt/proj3/eu-25-19/KITTI/kitti_infos_val.pkl \
	-validation_split_path /mnt/proj3/eu-25-19/KITTI/ImageSets/val.txt \
	-late_fusion_config $HOME/LCF3D/src/configs/example_cfg_clustering.json \
	-device cuda:0 \
	--view single
