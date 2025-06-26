import shutil
import time
import subprocess
import multiprocessing
import point_cloud_utils as pcu
from PIL import Image
from .generate_obj_pic import generate_meshs
from pathlib import Path
from multiprocessing import Pool
from .mylogging import Log

ROOT_PATH = Path(__file__).resolve().parent
BLENDER_MAIN_PROGRAM_PATH = ROOT_PATH / Path('../3rd/blender-4.2.2-linux-x64/blender')
BG_PLY_PATH = ROOT_PATH / Path('../static/bg.ply')
BLENDER_SCRIPT_TEMPLATE = (ROOT_PATH / Path('../static/blender_render_script_for_gif.py')).read_text()


def generate_blender_screenshot_from_meshs(meshs_root_path: Path, temp_file_path: Path, output_path: Path, scale, translation):
    template_path = temp_file_path / 'script.py'
    log_path = temp_file_path / 'log.txt'

    cur_script = (BLENDER_SCRIPT_TEMPLATE
            .replace("{{objs_path}}", meshs_root_path.as_posix())
            .replace("{{bg_ply_path}}", BG_PLY_PATH.as_posix())
            .replace("{{output_path}}", output_path.as_posix())
            .replace("{{r}}", '10')
            .replace("{{azimuth}}", '300')
            .replace("{{elevation}}", '30')
            .replace("{{USE_GPU}}", "True")
            .replace('"{{scale_default}}"', scale)
            .replace('"{{translation_default}}"', translation)
        )

    template_path.write_text(cur_script)

    start_time = time.time()
    with open(log_path.as_posix(), 'w') as log_file:
        process = subprocess.Popen([
                BLENDER_MAIN_PROGRAM_PATH.as_posix(),
                '--background',
                '--python', template_path.as_posix(),
            ],
            stdout=log_file, stderr=log_file
        )
        process.wait()

    if process.returncode != 0:
        Log.critical(f'{meshs_root_path} Blender failed with status {process.returncode}')
        exit(-1)
        return None, None
    else:
        Log.info(f'{meshs_root_path} Rendered in {time.time() - start_time:.2f}s with returncode = {process.returncode}')
        with open(log_path.as_posix(), 'r') as log_file:
            log_file = log_file.read()
            scale = log_file.split("SCALE'")[1].split("'")[0]
            translation = log_file.split("TRANSLATION'")[1].split("'")[0]
            return scale, translation


def generate_obj_pics_blender_batched_frame(_parts_data, percentage, workspace_path: Path, scale, translation):
    meshs, _ =  generate_meshs(_parts_data, percentage)
    mesh_path = workspace_path / "mesh"; mesh_path.mkdir()
    temp_path = workspace_path / "temp"; temp_path.mkdir()
    rest_path = workspace_path / "result.png"

    for idx, mesh in enumerate(meshs):
        # t0 = time.time()
        curr_mesh_path = mesh_path / f'{idx}.obj'
        mesh.export(curr_mesh_path)
        # v, f = pcu.load_mesh_vf(curr_mesh_path.as_posix())
        # v_eighth, f_eighth, _, _ = pcu.decimate_triangle_mesh(v, f, max_faces=f.shape[0] // 4)
        # pcu.save_mesh_vf(curr_mesh_path.as_posix(), v_eighth, f_eighth)
        # t1 = time.time()
        # print("Decimating Mesh time used: ", t1 - t0)


    scale, translation = generate_blender_screenshot_from_meshs(mesh_path, temp_path, rest_path, scale, translation)

    import shutil; shutil.rmtree(mesh_path, ignore_errors=True)

    buffer = Image.open(rest_path).convert('RGBA')
    buffer.info['disposal'] = 2
    return scale, translation, buffer



def generate_obj_pics_blender_batched(_parts_data, percentages, workspace_path: Path):
    init_path = workspace_path / "init"; init_path.mkdir(parents=True)

    # multiprocessing.set_start_method("spawn")

    scale, translation, _ = generate_obj_pics_blender_batched_frame(_parts_data, 1, init_path, "''", "''")
    Log.info(f"Use scale={scale}, translation={translation}")

    buffers = []
    with Pool(6) as pool:
        handler = []
        for percentage in percentages:
            temp_path = workspace_path / str(percentage)
            shutil.rmtree(temp_path, ignore_errors=True)
            temp_path.mkdir(parents=True, exist_ok=True)
            rt = pool.apply_async(generate_obj_pics_blender_batched_frame, (_parts_data, percentage, temp_path, scale, translation))
            handler.append(rt)

        for h in handler:
            _, _, buffer = h.get()
            buffers.append(buffer)

    return buffers
