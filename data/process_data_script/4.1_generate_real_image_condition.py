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
script = Path('<skip, ROOT_PATH_OF_PROJECT>/static/blender_render_script.template.py').read_text()
bg_ply_path = Path('<skip, ROOT_PATH_OF_PROJECT>/static/bg.ply')
blender_main_program_path = Path('<skip, ROOT_PATH_OF_PROJECT>/3rd/blender-4.2.2-linux-x64/blender')
n_png_per_obj = 10

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
    raw_dataset_path = Path('../datasets/0_raw_dataset')

    high_q_output_path = Path('../datasets/5_screenshot_real')
    shutil.rmtree(high_q_output_path, ignore_errors=True)
    high_q_output_path.mkdir(exist_ok=True)

    src_shapes_paths = []
    for raw_dataset in tqdm(list(raw_dataset_path.glob('*')), desc="Filting "):
        meta = json.loads((raw_dataset / 'meta.json').read_text())
        if meta['model_cat'] == 'StorageFurniture':
            src_shapes_paths.append(raw_dataset)

    for shape_path in tqdm(list(src_shapes_paths),
                         desc='Generating High-Quality Screenshots'):
        shape_name = f"StorageFurniture_{shape_path.stem}"
        Log.info(f"Processing {shape_name}")
        for _ in range(n_png_per_obj):
            generate_high_q_screenshot(shape_name, shape_path / "textured_objs", high_q_output_path)
            Log.info(f'[{shape_name}] Done, repeat = {_ + 1}')