export CUDA_VISIBLE_DEVICES=0 
python $HOME/HybridLateCascadeFusion/src/demo.py \
	-output_dir $HOME/HybridLateCascadeFusion/src/outputs \
	-img_path_left /DATA/kitti_mmdet3d/training/image_2/000001.png \
	-img_path_right /DATA/kitti_mmdet3d/training/image_3/000001.png \
    -lidar_path /DATA/kitti_mmdet3d/training/velodyne/000001.bin \
	-calib_path /DATA/kitti_mmdet3d/training/calib/000001.txt \
    -late_fusion_config $HOME/HybridLateCascadeFusion/src/example_cfg.json \
    -device cuda:0 \
    --visualize