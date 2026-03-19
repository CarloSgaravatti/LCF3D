from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import json
import mmcv
import tqdm
import pandas as pd
            
            
def kitti_annotations_to_coco(imgs_ids, category_mapping, dataset_root_path, 
                              save_file_path=None, iscrowd_dontcare=True):
    image_path = f'{dataset_root_path}/training/image_2'
    label_path = f'{dataset_root_path}/training/label_2'
    
    categories = []
    for name, category_id in category_mapping.items():
        categories.append({'id': category_id, 'name': name})
        
    annotation_id = 0
    coco_json = {'images': [], 'annotations': [], 'categories': categories}
    for image_id in tqdm.tqdm(imgs_ids):
        filename = f'{image_path}/{image_id}.png'
        image = mmcv.imread(filename)
        height, width = image.shape[:2]

        data_info = dict(file_name=f'{image_id}.png', width=width, height=height, id=int(image_id))
        coco_json['images'].append(data_info)

        # load annotations
        annotation_file = f'{label_path}/{image_id}.txt'
        with open(annotation_file, 'r') as annotations_file:
            lines = [image_id.strip('\n') for image_id in annotations_file.readlines()]

        content = [line.strip().split(' ') for line in lines]
        bbox_names = [x[0] for x in content]
        bboxes = [[float(info) for info in x[4:8]] for x in content]

        for bbox_name, bbox in zip(bbox_names, bboxes):
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            if bbox_name != 'DontCare':
                coco_json['annotations'].append({
                    'image_id': int(image_id),
                    'category_id': category_mapping[bbox_name],
                    'bbox': bbox,
                    'id': annotation_id,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0,
                })
                annotation_id += 1
            elif iscrowd_dontcare:
                # setting it to car but it can be any since it is ignored by mmdetection
                coco_json['annotations'].append({
                    'image_id': int(image_id),
                    'category_id': category_mapping['Car'],
                    'bbox': bbox,
                    'id': annotation_id,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 1,
                })
                annotation_id += 1
                
    if save_file_path is not None:
        with open(save_file_path, 'w') as annotation_file:
            json.dump(coco_json, annotation_file)
    return coco_json
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convert KITTI annotations to COCO format')
    parser.add_argument('--kitti_root_path', type=str, required=True, help='Path to the KITTI dataset root')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the COCO formatted annotations')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    kitti_root_path = args.kitti_root_path
    save_path = args.save_path

    class_mapping = {
        "Pedestrian": 1,
        "Cyclist": 2,
        "Car": 0,
        "Van": 3,
        "Truck": 4,
        "Misc": 5,
        "Tram": 6,
        "Person_sitting": 7
    }
    
    with open('/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/val.txt', 'r') as split_file:
        validation_ids = split_file.readlines()
    val_imgs = [val_id.rstrip('\n') for val_id in validation_ids] 
    
    with open('/mnt/datasets_1/carlos00/kitti_mmdet3d/ImageSets/train.txt', 'r') as split_file:
        training_ids = split_file.readlines()
    train_imgs = [train_id.rstrip('\n') for train_id in training_ids] 
    
    kitti_annotations_to_coco(train_imgs, class_mapping, kitti_root_path, os.path.join(save_path, 'train_annot_coco.json'))
    kitti_annotations_to_coco(val_imgs, class_mapping, kitti_root_path, os.path.join(save_path, 'val_annot_coco.json'))