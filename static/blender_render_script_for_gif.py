import os
import bpy
import math
import numpy as np
from mathutils import Vector

bpy.context.preferences.view.language = 'en_US'
bpy.context.preferences.view.use_translate_interface = True

resolution_x, resolution_y = (512, 512)
resolution_percentage = 100

# r, azimuth, elevation = (8, 0, 30)
r, azimuth, elevation = float('{{r}}'), float('{{azimuth}}'), float('{{elevation}}')

PI = 3.14159265357389

bright_color = ['#802222', '#91812f', '#3e1358', '#154f50', '#80a426', '#000080', '#8B4513', '#CD5C5C', '#222A35', '#2E54A1', '#331807']

def spherical_to_cartesian(r, azimuth, elevation):
    '''
        azimuth, elevation are in degree.
    '''
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    x = r * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = r * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = r * math.sin(elevation_rad)

    return x, y, z

class CameraParameters:
    location = Vector(spherical_to_cartesian(r, azimuth, elevation))
    lens = 200.0
    sensor_width = 36
    type = 'PERSP'

    @classmethod
    def apply(cls, camera):
        camera.location = cls.location
        camera.data.lens = cls.lens
        camera.data.sensor_width = cls.sensor_width
        camera.data.type = cls.type

def set_material(idx):
    obj = bpy.context.active_object

    color = bright_color[(idx + 0) % len(bright_color)]
    color_tuple = (int(color[1:3], 16) / 0xff,
                   int(color[3:5], 16) / 0xff,
                   int(color[5:7], 16) / 0xff, 1)

    if not obj.data.materials:
        mat = bpy.data.materials.new(name="NewMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color_tuple
        bsdf.inputs['Metallic'].default_value = 0.2
        bsdf.inputs['Roughness'].default_value = 0.7
        bsdf.inputs['Alpha'].default_value = 0.95

def focus_object(obj0, obj):
    '''
        make obj0 focus on obj
    '''
    obj_center = sum((Vector(b) for b in obj.bound_box), Vector()) / 8
    direction = obj_center - obj0.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj0.rotation_euler = rot_quat.to_euler()

def check_obj_bound(obj):
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_x = min(vertices, key=lambda v: v.x).x
    max_x = max(vertices, key=lambda v: v.x).x
    min_y = min(vertices, key=lambda v: v.y).y
    max_y = max(vertices, key=lambda v: v.y).y
    min_z = min(vertices, key=lambda v: v.z).z
    max_z = max(vertices, key=lambda v: v.z).z
    return min_x, max_x, min_y, max_y, min_z, max_z

def render_shape_blender(objs_path, bg_ply_path, output_path, use_gpu: bool, transparent_bg: bool):
    try: bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError as e: print(e)
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import objects from obj path
    obj_files = [f for f in os.listdir(objs_path) if f.endswith('.obj') or f.endswith('ply')]
    obj_files.sort()
    for idx, obj_file in enumerate(obj_files):
        obj_file_path = os.path.join(objs_path, obj_file)
        if obj_file_path.endswith('obj'):
            bpy.ops.wm.obj_import(filepath=obj_file_path)
        else:
            bpy.ops.wm.ply_import(filepath=obj_file_path)
        set_material(idx)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    # bpy.ops.object.shade_smooth()

    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')

    # First: scale
    min_x, max_x, min_y, max_y, min_z, max_z = check_obj_bound(obj)
    scale = 1.0 / max(max_x - min_x, max_y - min_y, max_z - min_z)
    print(f"SCALE'{scale}'")

    scale_default = "{{scale_default}}"
    if not isinstance(scale_default, str):
        scale = scale_default

    obj.scale *= scale
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(scale=True)

    # Second: translation
    min_x, max_x, min_y, max_y, min_z, max_z = check_obj_bound(obj)
    translation = Vector((-(min_x + max_x) / 2, -(min_y + max_y) / 2, -min_z))
    print(f"TRANSLATION'Vector(({-(min_x + max_x) / 2},{-(min_y + max_y) / 2},{-min_z}))'")

    translation_default = "{{translation_default}}"
    if not isinstance(translation_default, str):
        translation = translation_default

    obj.location += translation
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(location=True)

    if transparent_bg:
        # Make background transparent
        bpy.context.scene.render.film_transparent = True

    # Load Background Mesh.
    bpy.ops.wm.ply_import(filepath=bg_ply_path)
    bpy.ops.object.shade_smooth()

    # Add Materials To BG Mesh.
    bgmat = bpy.data.materials.new(name='bgmat')
    bgobj = bpy.context.active_object
    bgobj.data.materials.append(bgmat)
    bgmat.use_nodes = True
    edit_node = bgmat.node_tree.nodes["Principled BSDF"]
    edit_node.inputs['Base Color'].default_value = (1, 1, 1, 1) #(0.800204, 0.689428, 0, 1)
    edit_node.inputs['Metallic'].default_value = 0.444109
    edit_node.inputs['Roughness'].default_value = 1
    edit_node.distribution = 'GGX'
    edit_node.inputs['Emission Strength'].default_value = 2.0

    if transparent_bg:
        # Open Shadow Catcher, Only show its shadow.
        bpy.context.object.is_shadow_catcher = True

    # Add Lights
    ## Point Light
    bpy.ops.object.light_add(type='POINT', location=(1, 1, 2))
    light = bpy.context.active_object
    light.data.energy = 50
    focus_object(light, obj)

    ## Plain Light
    bpy.ops.object.light_add(type='AREA', location=(1, 1, 3))
    light = bpy.context.active_object
    light.data.energy = 300
    light.rotation_euler = (PI, 0, 0)

    # Add new camera
    bpy.ops.object.camera_add()
    new_cam = bpy.context.active_object
    CameraParameters.apply(new_cam)
    bpy.context.scene.camera = new_cam
    focus_object(new_cam, obj)

    # Render
    render = bpy.context.scene.render

    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = resolution_percentage
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    render.image_settings.file_format = 'PNG'
    render.filepath = output_path

    if use_gpu:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()

    bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    objs_path = "{{objs_path}}"
    bg_ply_path = "{{bg_ply_path}}"
    output_path = "{{output_path}}"

    render_shape_blender(objs_path, bg_ply_path, output_path, {{USE_GPU}}, True)