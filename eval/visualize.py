# [ArtFormer]: use blender to render the List[trimesh.Trimesh] to a png file

import time
import pickle
import shutil
import trimesh
import subprocess
from pathlib import Path
import pyvista as pv
import point_cloud_utils as pcu

import sys
sys.path.append('..')
from utils.generate_obj_pic import generate_meshs
from utils.mylogging import Log

SMOOTH_MESH_ITER_NUM = 1
BLENDER_MAIN_PROGRAM_PATH = Path('3rd/blender-4.2.2-linux-x64/blender')
BG_PLY_PATH = Path('static/bg.ply')
BLENDER_SCRIPT_TEMPLATE = Path('static/blender_render_script_figure.template.py').read_text()
USE_GPU = True

def smooth_mesh(src: Path, dist: Path):
    v, f = pcu.load_mesh_vf(src.as_posix())
    v_smooth = pcu.laplacian_smooth_mesh(v, f,
                                         num_iters=SMOOTH_MESH_ITER_NUM)
    pcu.save_mesh_vf(dist.as_posix(), v_smooth, f)
    Log.info("[Write] %s", dist)

def generate_simple_screenshot_from_meshs(meshs_path: list[Path], output_path: Path):
    plotter = pv.Plotter(off_screen=True)
    for idx, mesh_path in enumerate(meshs_path):
        mesh = trimesh.load_mesh(mesh_path.as_posix())
        if len(mesh.vertices) <= 3:
            Log.warning("length of mesh vertices %s, ignore. path: %s", len(mesh.vertices), mesh_path)
            continue
        plotter.add_mesh(mesh, color=['#E9A7AB', '#F5D76C', '#EB950C', '#DB481F', '#08998A', '#FF2D2B'][idx % 6])

    plotter.add_axes()
    plotter.camera_position = [(3.487083128152961, 1.8127192062148014, 1.9810015800028038), (-0.04570716149497277, -0.06563260832821388, -0.06195879116203942), (-0.37480300238091124, 0.9080915656577206, -0.18679512249404312)]
    plotter.show()
    buffer = plotter.screenshot(output_path.as_posix())
    Log.info("[Write] %s", output_path)
    plotter.close()

    return buffer

def generate_blender_screenshot_from_meshs(meshs_root_path: Path, temp_file_path: Path, output_path: Path):
    template_path = temp_file_path / 'script.py'
    log_path = temp_file_path / 'log.txt'

    cur_script = (BLENDER_SCRIPT_TEMPLATE
            .replace("{{objs_path}}", meshs_root_path.as_posix())
            .replace("{{bg_ply_path}}", BG_PLY_PATH.as_posix())
            .replace("{{output_path}}", output_path.as_posix())
            .replace("{{r}}", '10')
            .replace("{{azimuth}}", '300')
            .replace("{{elevation}}", '30')
            .replace("{{USE_GPU}}", "True" if USE_GPU else "False")
        )

    template_path.write_text(cur_script)

    start_time = time.time()
    with open(log_path.as_posix(), 'w') as log_file:
        process = subprocess.Popen([
                BLENDER_MAIN_PROGRAM_PATH.as_posix(),
                '--background',
                '--python', template_path.as_posix(),
            ]
            , stdout=log_file, stderr=log_file
            )
        process.wait()

    if process.returncode != 0:
        Log.critical(f'{meshs_root_path} Blender failed with status {process.returncode}')
        exit(-1)
    Log.info(f'{meshs_root_path} Rendered in {time.time() - start_time:.2f}s with returncode = {process.returncode}')

def visualize_obj_high_q(obj_data, temp_output_path: Path, output_path: Path, percentage):
    shutil.rmtree(temp_output_path, ignore_errors=True)
    (temp_output_path / "raw").mkdir(parents=True, exist_ok=True)
    (temp_output_path / "bbx").mkdir(parents=True, exist_ok=True)

    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    mesh_pair = generate_meshs(obj_data, percentage)
    meshs : list[trimesh.Trimesh]       = mesh_pair[0]
    bbox_meshs : list[trimesh.Trimesh]  = mesh_pair[1]

    assert len(meshs) == len(bbox_meshs)

    for idx in range(len(meshs)):
        mesh = meshs[idx]
        Log.info("length of mesh vertices %s", len(mesh.vertices))
        if len(mesh.vertices) <= 3:
            Log.warning("length of mesh vertices %s, ignore. idx = %s", len(mesh.vertices), idx)
            continue

        raw_mesh_output_path = temp_output_path / "raw" / f"{idx}.obj"
        mesh.export(raw_mesh_output_path)
        Log.info("[Write] %s", raw_mesh_output_path)

        bbox_mesh = bbox_meshs[idx]
        bbx_mesh_output_path = temp_output_path / "bbx" / f"{idx}.obj"
        bbox_mesh.export(bbx_mesh_output_path)
        Log.info("[Write] %s", bbx_mesh_output_path)

        # smo_mesh_output_path = temp_output_path / "smo" / f"{idx}.obj"
        # smooth_mesh(raw_mesh_output_path, smo_mesh_output_path)

    # Log.info("Runnning Simple Script %s", temp_output_path / "smo")
    # generate_simple_screenshot_from_meshs(list((temp_output_path / "smo").glob('*')), output_path / "simple-smo.png")

    # Log.info("Runnning Simple Script %s", temp_output_path / "raw")
    # generate_simple_screenshot_from_meshs(list((temp_output_path / "raw").glob('*')), output_path / "simple-raw.png")

    # !!!![ArtFormer]: TEMP CHANGE JUST FOR DEBUG !!!!!
    # Log.info("Runnning Simple Script %s", temp_output_path / "raw")
    # generate_simple_screenshot_from_meshs(list((temp_output_path / "raw").glob('*')), output_path / "result.png")
    Log.info("Runnning Blender Script %s", temp_output_path / "raw")
    generate_blender_screenshot_from_meshs(temp_output_path / "raw", temp_output_path, output_path / "result.png")

    Log.info("Runnning Blender Script %s", temp_output_path / "bbx")
    generate_blender_screenshot_from_meshs(temp_output_path / "bbx", temp_output_path, output_path / "result-bbx.png")


if __name__ == '__main__':
    temp_output_path = Path('log') / ("temp_visual_" + time.strftime("%m-%d-%I%p-%M-%S"))
    filename = '<skip>'

    data = pickle.load(open(filename, "rb"))

    for idx, d in enumerate(data):
        visualize_obj_high_q(d['data'], temp_output_path / "temp" / (str(idx) + '_0'), temp_output_path / "result" / (str(idx) + '_0'), 0)
        visualize_obj_high_q(d['data'], temp_output_path / "temp" / (str(idx) + '_1'), temp_output_path / "result" / (str(idx) + '_1'), 1)
