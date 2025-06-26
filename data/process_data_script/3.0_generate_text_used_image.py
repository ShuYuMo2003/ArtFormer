import json
import time
import shutil
import random
import subprocess
import trimesh
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append('../..')
from utils.mylogging import console, Log
from utils import generate_random_string

script = Path('../../static/blender_render_script_figure.template.py').read_text()
bg_ply_path = Path('../../static/bg.ply')
blender_main_program_path = Path('../../3rd/blender-4.2.2-linux-x64/blender')
n_png_per_obj = 1

random.seed(20031006)

def generate_high_q_screenshot(obj_name: str, textured_obj_path: Path, output_dir: Path):
    suffix = generate_random_string(4)
    template_path = output_dir / f'script-{obj_name}-{suffix}.py'
    log_path = output_dir / f'log-{obj_name}-{suffix}.log'

    cur_script = (script
            .replace("{{objs_path}}", textured_obj_path.as_posix())
            .replace("{{bg_ply_path}}", bg_ply_path.as_posix())
            .replace("{{output_path}}", (output_dir / (obj_name + f'-{suffix}.png')).as_posix())
            .replace("{{r}}", f'8')
            .replace("{{azimuth}}", f'300')
            .replace("{{elevation}}", f'30')
            .replace("{{USE_GPU}}", "True")
        )

    template_path.write_text(cur_script)

    start_time = time.time()
    with open(log_path.as_posix(), 'w') as log_file:
        process = subprocess.Popen([
                blender_main_program_path.as_posix(),
                '--background',
                # '--cycles-device', 'CUDA'
                '--python', template_path.as_posix(),
            ]
            , stdout=log_file, stderr=log_file
            )
        process.wait()

    if process.returncode != 0:
        Log.critical(f'[{obj_name}] Blender failed with status {process.returncode}')
        exit(-1)
    Log.info(f'[{obj_name}] Rendered in {time.time() - start_time:.2f}s with returncode = {process.returncode}')

if __name__ == '__main__':
    meshs_path = Path('../datasets/1_preprocessed_mesh')

    processed_mesh_path = Path('../datasets/1.5_preprocessed_obj_mesh')
    high_q_output_path = Path('../datasets/4_screenshot_high_q')
    shutil.rmtree(processed_mesh_path, ignore_errors=True)
    shutil.rmtree(high_q_output_path, ignore_errors=True)

    # need_stem = [
    #     'StorageFurniture_45984',
    #     'StorageFurniture_45238',
    #     'StorageFurniture_45168',
    # ]

    high_q_output_path.mkdir(exist_ok=True)

    extract_info = json.loads((Path('../datasets') / 'meta.json').read_text())['1_extract_from_raw_dataset']
    train_keys = extract_info['train_split'] + extract_info['test_split']

    shape_mesh = {}

    for mesh_path in tqdm(list(meshs_path.glob('*')),
                     desc="Prefetch mesh for each shape"):
        mesh_info = mesh_path.stem.split('_')
        shape_name = '_'.join(mesh_info[:2])
        if shape_mesh.get(shape_name) is None:
            shape_mesh[shape_name] = [mesh_path]
        else:
            shape_mesh[shape_name].append(mesh_path)

    shape_name_2_mesh_path = {}
    for shapes_name in tqdm(list(shape_mesh.keys()),
                     desc="Prefetch mesh for each shape and save"):
        shape_mesh_output_path : Path = processed_mesh_path / shapes_name
        shape_mesh_output_path.mkdir(exist_ok=True, parents=True)
        num = len(sorted(shape_mesh[shapes_name]))
        for idx, part_mesh_path in enumerate(sorted(shape_mesh[shapes_name])):
            mesh = trimesh.load_mesh(part_mesh_path.as_posix())
            mesh.export(shape_mesh_output_path / f"{num - idx - 1}.obj")

        shape_name_2_mesh_path[shapes_name] = shape_mesh_output_path

    for shapes_name in tqdm(list(shape_name_2_mesh_path.keys()),
                         desc='Generating High-Quality Screenshots'):
        Log.info(f"Processing {shapes_name}")

        # if shapes_name not in need_stem: continue

        print(shapes_name)

        for _ in range(n_png_per_obj):
            generate_high_q_screenshot(shapes_name, shape_name_2_mesh_path[shapes_name], high_q_output_path)
            Log.info(f'[{shapes_name}] Done, repeat = {_ + 1}')