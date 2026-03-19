from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import torch
from typing_extensions import Union, Dict, Tuple, List, Set
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes, LiDARInstance3DBoxes, CameraInstance3DBoxes
from bbox_utils import corners_to_axis_aligned_bbox, corners_to_img_coord
from matplotlib import cm
from colorsys import hsv_to_rgb, rgb_to_hsv

def draw_bboxes_3d_image(image: np.ndarray, 
                         bboxes_3d: Union[np.ndarray, torch.Tensor, BaseInstance3DBoxes], 
                         labels: Union[np.ndarray, torch.Tensor, List],
                         projection_matrix: torch.Tensor, 
                         lidar_to_cam: torch.Tensor,
                         color_dict: Dict[int, Tuple],
                         lidar_coords: bool = True,
                         save_path: Union[Path, str] = None):
    if isinstance(bboxes_3d, np.ndarray) or isinstance(bboxes_3d, torch.Tensor):
        bboxes_3d = LiDARInstance3DBoxes(bboxes_3d) if lidar_coords else CameraInstance3DBoxes(bboxes_3d)
        
    img_shape = image.shape
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    corners_3d = bboxes_3d.corners
    corners_image = corners_to_img_coord(corners_3d, projection_matrix, lidar_coords, lidar_to_cam)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    for i, (corners, label) in enumerate(zip(corners_image.numpy(), labels)):
        
        # do not visualize bboxes 3d that do not have any projected corners inside the image
        is_outside = (corners[:, 0] < 0) | (corners[:, 1] < 0) | \
            (corners[:, 0] > img_shape[1]) | (corners[:, 1] > img_shape[0])
        
        if np.sum(~is_outside) == 0:
            continue

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines connecting top and bottom faces
        ]

        for edge in edges:
            start_point = tuple(corners[edge[0]])
            end_point = tuple(corners[edge[1]])
            draw.line([start_point, end_point], fill=color_dict[label], width=1)
            
    if save_path:
        image.save(save_path)
            
    return image


def draw_bboxes_2d(image: np.array, bboxes: np.array, labels: np.array, classes_dict: Dict[int, str],
                   scores: np.array = None, save_path: str = None, color_dict: Dict[int, Tuple] = None, 
                   fill: bool = False, alpha: int = 50) -> None:
    '''
    Draw bounding boxes in xyxy format in the specified image
    '''
    if fill:
        rgba_image_array = np.concatenate([image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        image = Image.fromarray(rgba_image_array)
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(image)
        image_draw_overlay = ImageDraw.Draw(overlay, 'RGBA')
    else:
        image = Image.fromarray(image)
        image_draw = ImageDraw.Draw(image)
        
    tabu_set = set()
    texts = []
    num_classes = len(classes_dict.keys())
    
    if color_dict is None:
        color_dict = {}
        fill_dict = {}
        for label in classes_dict.keys():
            color_dict[label] = cmap_color(label, num_classes)
            fill_dict[label] = cmap_color(label, num_classes, alpha=alpha) if fill else None
    else:
        fill_dict = {}
        for label in classes_dict.keys():
            fill_dict[label] = color_dict[label] + (50,) if fill else None
        
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if label not in classes_dict:
            # If the sample is from background or padding ignore it
            continue
        x_0 = bbox[0]
        y_0 = bbox[1]
        x_1 = bbox[2]
        y_1 = bbox[3]
        if fill:
            image_draw_overlay.rectangle((x_0, y_0, x_1, y_1), outline=None, fill=fill_dict[label], width=1)
        image_draw.rectangle((x_0, y_0, x_1, y_1), outline=color_dict[label], fill=None, width=1)
        point, tabu_points = find_feasible_label((y_0, x_0, y_1, x_1), tabu_set, (image.height, image.width))
        text = classes_dict[label] if scores is None else f"{classes_dict[label]} {100*scores[i]:.2f}"
        texts.append({"xy": (point[0], point[1]), "text": text,
                      "fill": color_dict[label]})
        tabu_set = tabu_set.union(tabu_points)

    # Apply text at last to avoid boxes covering it
    for text in texts:
        image_draw.text(*text.values())
       
    if fill:
        image = Image.alpha_composite(image, overlay)

    if save_path:
        image.save(save_path)
        
    return image
    
    
def cmap_color(i: int, n: int, cmap: cm = cm.rainbow, alpha: int = 255,
               desaturate: bool = False) -> Tuple[int, ...]:
    color = cmap(i / n)[:3]
    if desaturate:
        h, s, v = rgb_to_hsv(*color)
        color = hsv_to_rgb(h, s / 2, v)
    return tuple([*(int(c * 255) for c in color), alpha])


def find_feasible_label(box: Union[Tuple[int, int, int, int], List[int]], tabu_set: Set[Tuple[int]], fig_shape: Tuple[int, int],
                        label_width: int = 50, label_height: int = 20) -> Tuple[Tuple[int, int], Set[Tuple[int, int]]]:

    y_top = int(box[0])
    x_left = int(box[1])
    y_bottom = int(box[2])
    x_right = int(box[3])

    # Points
    points = {
        1: (x_right + 2, y_top),
        2: (x_right + 2, y_bottom - label_height),
        3: (x_right - label_width, y_bottom + 2),
        4: (x_left, y_bottom + 2),
        5: (x_left - label_width, y_bottom - label_height),
        6: (x_left - label_width, y_top),
        7: (x_left, y_top - label_height),
        8: (x_right - label_width, y_top - label_height)
    }

    # Feasibility according to image boundaries
    feasible = {
        1: (x_right + label_width <= fig_shape[1]) and (y_top + label_height <= fig_shape[0]),
        2: (x_right + label_width <= fig_shape[1]) and (y_bottom - label_height >= 0),
        3: (x_right - label_width >= 0) and (y_bottom + label_height <= fig_shape[0]),
        4: (x_left + label_width <= fig_shape[1]) and (y_bottom + label_height <= fig_shape[0]),
        5: (x_left - label_width >= 0) and (y_bottom - label_height >= 0),
        6: (x_left - label_width >= 0) and (y_top + label_height <= fig_shape[0]),
        7: (x_left + label_width <= fig_shape[1]) and (y_top - label_height >= 0),
        8: (x_right - label_width >= 0) and (y_top - label_height >= 0)
    }

    tabu_points = {}
    # Return the first feasible point not in the tabu list
    for position, point in points.items():
        tabu_points[position] = {(x_tabu, y_tabu) for x_tabu in range(point[0], point[0] + label_width) for y_tabu in
                                 range(point[1], point[1] + label_height)}
        if feasible[position] and len(tabu_points[position].intersection(tabu_set)) == 0:
            return point, tabu_points[position]

    # If none was feasible and not overlapping, relaxes the non-overlapping constraint 
    # and returns the first feasible point
    position = [position for position, f in feasible.items() if f][0]
    return points[position], tabu_points[position]