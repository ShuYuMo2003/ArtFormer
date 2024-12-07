# [ArtFormer:] This file is adapted from https://github.com/3dlg-hcvc/cage/blob/eb8d8155905883ba7c6ce235fa989807733f7d11/utils/render.py#L1

import os
import trimesh
import pyrender
import numpy as np
import open3d as o3d
from copy import deepcopy
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_rotation_axis_angle(k, theta):
    '''
    Rotation matrix converter from axis-angle using Rodrigues' rotation formula

    Args:
        k (np.ndarray): 3D unit vector representing the axis to rotate about.
        theta (float): Angle to rotate with in radians.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    '''
    if np.linalg.norm(k) == 0.:
        return np.eye(3)
    k = k / np.linalg.norm(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def get_bbox_mesh_pair(center, size, axis_o, axis_d, radius=0.01, without_axis=False):
    '''
    Function to get the bounding box mesh pair

    Args:
    - center (np.array): bounding box center
    - size (np.array): bounding box size
    - axis_d (np.array): axis direction
    - axis_o (np.array): axis origin
    - radius (float): radius of the cylinder

    Returns:
    - trimesh_box (trimesh object): trimesh object for the bbox at resting state
    '''

    size = np.clip(size, a_max=3, a_min=0.005)
    center = np.clip(center, a_max=3, a_min=-3)

    line_box = o3d.geometry.TriangleMesh()
    z_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[2])
    y_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[1])
    R_y = get_rotation_axis_angle(np.array([1., 0., 0.]), np.pi / 2)
    y_cylinder.rotate(R_y, center=(0, 0, 0))
    x_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size[0])
    R_x = get_rotation_axis_angle(np.array([0., 1., 0.]), np.pi / 2)
    x_cylinder.rotate(R_x, center=(0, 0, 0))


    z1 = deepcopy(z_cylinder)
    z1.translate(np.array([-size[0] / 2, size[1] / 2, 0.]))
    line_box += z1.translate(center[:3])
    z2 = deepcopy(z_cylinder)
    z2.translate(np.array([size[0] / 2, size[1] / 2, 0.]))
    line_box += z2.translate(center[:3])
    z3 = deepcopy(z_cylinder)
    z3.translate(np.array([-size[0] / 2, -size[1] / 2, 0.]))
    line_box += z3.translate(center[:3])
    z4 = deepcopy(z_cylinder)
    z4.translate(np.array([size[0] / 2, -size[1] / 2, 0.]))
    line_box += z4.translate(center[:3])

    y1 = deepcopy(y_cylinder)
    y1.translate(np.array([-size[0] / 2, 0., size[2] / 2]))
    line_box += y1.translate(center[:3])
    y2 = deepcopy(y_cylinder)
    y2.translate(np.array([size[0] / 2, 0., size[2] / 2]))
    line_box += y2.translate(center[:3])
    y3 = deepcopy(y_cylinder)
    y3.translate(np.array([-size[0] / 2, 0., -size[2] / 2]))
    line_box += y3.translate(center[:3])
    y4 = deepcopy(y_cylinder)
    y4.translate(np.array([size[0] / 2, 0., -size[2] / 2]))
    line_box += y4.translate(center[:3])

    x1 = deepcopy(x_cylinder)
    x1.translate(np.array([0., -size[1] / 2, size[2] / 2]))
    line_box += x1.translate(center[:3])
    x2 = deepcopy(x_cylinder)
    x2.translate(np.array([0., size[1] / 2, size[2] / 2]))
    line_box += x2.translate(center[:3])
    x3 = deepcopy(x_cylinder)
    x3.translate(np.array([0., -size[1] / 2, -size[2] / 2]))
    line_box += x3.translate(center[:3])
    x4 = deepcopy(x_cylinder)
    x4.translate(np.array([0., size[1] / 2, -size[2] / 2]))
    line_box += x4.translate(center[:3])

    def get_axis_mesh(k, axis_o):
        '''
        Function to get the axis mesh

        Args:
        - k (np.array): axis direction
        - axis_o (np.array): axis origin
        '''

        k = k / np.linalg.norm(k)

        axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.025, cone_radius=0.04, cylinder_height=1.0, cone_height=0.08)
        arrow = np.array([0., 0., 1.], dtype=np.float32)
        n = np.cross(arrow, k)
        rad = np.arccos(np.dot(arrow, k))
        R_arrow = get_rotation_axis_angle(n, rad)
        axis.rotate(R_arrow, center=(0, 0, 0))
        axis.translate(axis_o[:3])
        axis.compute_vertex_normals()
        # vertices = np.asarray(axis.vertices)
        # faces = np.asarray(axis.triangles)
        # trimesh_axis = trimesh.Trimesh(vertices=vertices, faces=faces)
        # trimesh_axis.visual.vertex_colors = np.array([0.5, 0.5, 0.5, 1.0])
        return axis

    if not without_axis:
        axis_mesh = get_axis_mesh(axis_d, axis_o)
        line_box += axis_mesh

    vertices = np.asarray(line_box.vertices)
    faces = np.asarray(line_box.triangles)
    trimesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_box.visual.vertex_colors = np.array([0.0, 1.0, 1.0, 1.0])

    return trimesh_box




