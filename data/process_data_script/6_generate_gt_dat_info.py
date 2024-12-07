import json
import time
import shutil
import trimesh
import pickle
import numpy as np
from rich import print
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import point_cloud_utils as pcu

transformer_dataset_path = Path('../datasets/4_transformer_dataset')

output_path = Path('../datasets/5_obj_dat')
temp_path = output_path / 'temp'
shutil.rmtree(output_path, ignore_errors=True)
temp_path.mkdir(exist_ok=True, parents=True)
num_sample_points = 1000000

needed_category = [
    'StorageFurniture'
]

def mesh_to_point(mesh_path: Path, mesh_name: str):
    s_time = time.time()
    print("Generate sampling point for ", mesh_path)

    mesh_obj_path = temp_path / (mesh_name + '.obj')

    mesh = trimesh.load_mesh(mesh_path)
    (min_bound, max_bound) = mesh.bounds
    mesh.export(mesh_obj_path)

    v, f = pcu.load_mesh_vf(mesh_obj_path)
    resolution = 13_000
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    points = np.random.uniform(low=min_bound, high=max_bound, size=(num_sample_points, 3))

    sdf, fid, bc = pcu.signed_distance_to_mesh(points, vw, fw)

    inside_point = points[sdf < 0]

    rho = num_sample_points / (max_bound - min_bound).prod()

    print("rho =", rho, "inside_point shape =", inside_point.shape[0], "time used =", time.time() - s_time, " inside_ratio =", inside_point.shape[0] / num_sample_points)

    return inside_point, rho


def process_v_list(k, v_list):
    for idx, v in enumerate(v_list):
        del v['name']
        del v['latent_code']
        del v['text_hat']
        v['point'], v['rho'] = mesh_to_point(Path('../datasets/1_preprocessed_mesh') / v['mesh'], k + '_' + str(idx))

    with open(output_path / f'{k}.dat', 'wb') as f:
        f.write(pickle.dumps(v_list))

    #     just for clear memory.
    for v in v_list:
        del v['point']


def main():
    shape_name_to_info = {}
    for json_path in tqdm(list(transformer_dataset_path.glob('*.json')), desc="FLAG 1"):
        if json_path.stem == 'meta': continue
        shape_name = '_'.join(json_path.stem.split('_')[:2])
        category = json_path.stem.split('_')[0]
        if category not in needed_category: continue

        json_content = json.loads(json_path.read_text())
        shape_name_to_info[shape_name] = json_content['shape_info']

    with Pool(16) as pool:
        result = []
        for k, v_list, in tqdm(shape_name_to_info.items(), desc="FLAG 2"):
            result.append(pool.apply_async(process_v_list, (k, v_list)))

        bar = tqdm(total=len(result), desc='Mesh to Point')
        while result:
            for r in result:
                if r.ready():
                    bar.update(1)
                    result.remove(r)
            time.sleep(0.1)



if __name__ == '__main__':
    main()