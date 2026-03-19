from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d
from typing_extensions import Union, Dict, Tuple
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes, LiDARInstance3DBoxes, CameraInstance3DBoxes


def show_bboxes_3d(pc_path, lidar_to_cam, bboxes_3d: LiDARInstance3DBoxes, labels, 
                   colors_dict={0: (1, 1, 0), 1: (0, 1, 0), 2: (1, 0, 0)}, 
                   window_name='Open3D', save_path=None, save_capture=False, show_window=True, 
                   adjust_position=True, fill_points_inside=True):
    scan = np.fromfile(pc_path, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3]
    velo = np.insert(points, 3, 1, axis=1).T

    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.open3d.utility.Vector3dVector(points[:, :3])
    
    if fill_points_inside:
        point_labels = bboxes_3d.to('cuda').points_in_boxes_part(torch.tensor(points, device='cuda')).cpu().numpy()
        pcd.colors = open3d.open3d.utility.Vector3dVector([colors_dict[labels[l]] if l != -1 else (1, 1, 1) for l in point_labels])
    pcd.colors = open3d.open3d.utility.Vector3dVector([(1, 1, 1)] * points.shape[0])

    bboxes_linesets = []
    corners_pc = []

    corners_3d = bboxes_3d.corners
    for i, corners in enumerate(corners_3d.numpy()):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),   # Vertical lines connecting top and bottom faces
        ]
        
        line_mesh = LineMesh(corners, edges, [colors_dict[labels[i]] for _ in range(len(edges))], radius=0.02)
        bboxes_linesets.append(line_mesh)

        corners_draw = open3d.open3d.geometry.PointCloud()
        corners_draw.points = open3d.open3d.utility.Vector3dVector(corners)
        corners_draw.paint_uniform_color(colors_dict[labels[i]])
        # corners_draw.size = 10
        corners_pc.append(corners_draw)
    
    coor_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=1080)
    vis.add_geometry(coor_frame)
    vis.add_geometry(pcd)
    for bb_lineset, corners in zip(bboxes_linesets, corners_pc):
        bb_lineset.add_line(vis)
        # vis.add_geometry(bb_lineset)
        vis.add_geometry(corners)
        
    vis.get_render_option().background_color = [0, 0, 0]
    
    vis.get_render_option().point_size = 2  # Larger size for points inside the frustum
    vis.get_render_option().line_width = 10  # Adjust as needed
    
    vis.reset_view_point(True)
    vis.get_view_control().set_up((0, 0, 1))
    vis.get_view_control().set_front((1, 0, 0))
    # vis.get_view_control().set_lookat((0, 0, 0))
    vis.get_view_control().set_zoom(1.0) 

    # default camera parameters
    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    
    if adjust_position:
        extrinsic = lidar_to_cam.clone().numpy()

        # Translate the camera upwards slightly (along Z-axis)
        extrinsic[1, 3] += 12 # Adjust this value based on how much you want to move the camera up
        extrinsic[2, 3] += 4

        # Rotate the camera downwards (pitch rotation around the X-axis)
        angle = np.radians(35)  # 10 degrees downward tilt, adjust as needed
        # rotation_matrix = np.array([
        #     [np.cos(angle), 0, np.sin(angle), 0],
        #     [0, 1, 0, 0],
        #     [-np.sin(angle), 0, np.cos(angle), 0],
        #     [0, 0, 0, 1]
        # ])
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        extrinsic = np.dot(rotation_matrix, extrinsic)

        camera_params.extrinsic = extrinsic
        # camera_params.extrinsic = calibration_data['Tr_velo_to_cam'].numpy()
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, True)
    
    # vis.update_geometry()
    # vis.poll_events()
    vis.update_renderer()
    
    o3d_screenshot_mat = None
    if save_capture:
        o3d_screenshot_mat = vis.capture_screen_float_buffer(True)
        o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        Image.fromarray(o3d_screenshot_mat , "RGB").save(save_path)
    
    if show_window:
        vis.run()
    else:
        vis.destroy_window()
        
    return o3d_screenshot_mat


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = open3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=open3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)