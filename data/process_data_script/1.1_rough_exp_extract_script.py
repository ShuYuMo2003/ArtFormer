# experimental code.
# Handle only `Bottle``. Separate `lid`/`closure` and leave the rest of the part as another part.
import json
import time
import shutil
from tqdm import tqdm
from multiprocessing import Pool
from rich import print
import numpy as np
# import open3d as o3d
import pyvista as pv
from pathlib import Path
import point_cloud_utils as pcu

def find_obj(obj, result):
    objs_file = set(obj.get('objs', []))
    for child in obj.get('children', []):
        objs = find_obj(child, result)
        objs_file = objs_file | objs
    result[obj['name']] = list(objs_file)
    return objs_file

def merge_meshs(meshs_path: list[Path]) -> any:
    meshs = []
    for mesh_path in meshs_path:
        try:
            value = pv.read(str(mesh_path))
            meshs.append(value)
        except FileNotFoundError:
            print(f"[Warning] {mesh_path} not found.")

    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    return merged_mesh

def save_save(mesh_path, output_path):
    v, f = pcu.load_mesh_vf(str(mesh_path))
    # resolution = 12_000
    # vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    # points = np.asarray(vw)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # obb = pcd.get_oriented_bounding_box()

    # center = obb.center
    # R = obb.R
    # extent = obb.extent / 2 # for fit into [-1, -1, -1] [1, 1, 1]

    # _vw = (np.linalg.inv(R) @ (vw - center).T).T / extent

    pcu.save_mesh_vf(str(output_path), v, f)

    # result =  {
    #     'center': center.tolist(),
    #     'R': R.tolist(),
    #     'extent': extent.tolist()
    # }

    # return result

def process(shape_path: Path, output_info_path, output_mesh_path):
    shape_id = shape_path.stem

    result_file = json.loads((shape_path / 'result.json').read_text())
    meta = json.loads((shape_path / 'meta.json').read_text())
    key_name = f"{meta['model_cat']}_{meta['anno_id']}"

    assert len(result_file) == 1
    name_to_objs = {}
    find_obj(result_file[0], name_to_objs)

    part0_name = None
    for key in ['lid', 'closure', 'mouth']:
        if key in name_to_objs.keys():
            part0_name = key

    # print(name_to_objs.keys())
    # print(shape_path)

    if part0_name is None:
        print('!!!', 'Error', shape_path, name_to_objs.keys())
        return None

    all_objs = {obj for objs in name_to_objs.values() for obj in objs}

    part_a_objs = set(name_to_objs[part0_name])  # lid
    part_b_objs = all_objs - part_a_objs    # body

    mesh_a = merge_meshs([shape_path / 'objs' / (name + '.obj') for name in part_a_objs])
    mesh_b = merge_meshs([shape_path / 'objs' / (name + '.obj') for name in part_b_objs])

    mesh_a.save(output_mesh_path / (key_name + "_0.ply"))
    mesh_b.save(output_mesh_path / (key_name + "_1.ply"))

    (output_info_path / (key_name + ".json")).write_text(json.dumps({
        'meta': meta,
        'part': [
            {
                'name': 'lid',
                'mesh': (key_name + "_0.ply")
            },
            {
                'name': 'body',
                'mesh': (key_name + "_1.ply")
            }
        ]
    }))

    # save_save(*([output_mesh_path / (key_name + "_0.ply")] * 2))
    # save_save(*([output_mesh_path / (key_name + "_1.ply")] * 2))

    print(key_name, shape_path.stem, 'done')


if __name__ == '__main__':
    ex_raw_data_path = Path('../datasets/0_raw_partnet_dataset')
    output_info_path    = Path('../datasets/1_preprocessed_info/ex')
    output_mesh_path    = Path('../datasets/1_preprocessed_mesh/ex')

    ex_raw_data_paths = list(ex_raw_data_path.glob('*'))

    shutil.rmtree(output_info_path, ignore_errors=True)
    shutil.rmtree(output_mesh_path, ignore_errors=True)

    output_info_path.mkdir(parents=True, exist_ok=True)
    output_mesh_path.mkdir(parents=True, exist_ok=True)

    # for shape_path in ex_raw_data_paths:
    #     process(shape_path, output_info_path, output_mesh_path)

    with Pool(5) as p:
        results = [
            p.apply_async(process, (shape_path, output_info_path, output_mesh_path))
            for shape_path in tqdm(ex_raw_data_paths)
        ]

        bar = tqdm(total=len(ex_raw_data_paths), desc='Converting meshes')
        while results:
            for r in results:
                if not r.ready(): continue
                bar.update(1)
                status = r.get()
                results.remove(r)

            time.sleep(0.1)