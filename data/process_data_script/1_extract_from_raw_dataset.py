import json
import shutil
import time
# import open3d as o3d
import numpy as np
import pyvista as pv
from tqdm import tqdm
from rich import print
import random
from glob import glob
from pathlib import Path
import point_cloud_utils as pcu
from multiprocessing import Pool

import sys
sys.path.append('../..')
from utils import HighPrecisionJsonEncoder


def degree2rad(degree):
    # assert -180 <= degree <= 180
    return degree * np.pi / 180

def parse_partid_to_objs(shape_path:Path):
    result_file_path = shape_path / 'result.json'
    result_file = json.loads(result_file_path.read_text())
    partid_to_objs = {}
    def parse_part(part):
        pid = part['id']
        partid_to_objs[pid] = set(part.get('objs', set()))
        for child in part.get('children', []):
            parse_part(child)
            childs_objs = partid_to_objs[child['id']]
            partid_to_objs[pid] |= childs_objs

    assert len(result_file) == 1
    parse_part(result_file[0])

    return partid_to_objs

def merge_meshs(meshs_paths: list[Path], output_path:Path):
    meshs_paths = list(set(meshs_paths))
    meshs = []
    for mesh_path in meshs_paths:
        try:
            value = pv.read(str(mesh_path))
            meshs.append(value)
        except FileNotFoundError:
            print(f"[Warning] {mesh_path} not found.")

    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    merged_mesh.save(str(output_path))
    return merged_mesh

def parse_limit(part_data):
    if part_data['parent'] == -1 or part_data['joint'] in ['junk', 'fixed']:
        return [0., 0., 0., 0.]
    limit = part_data['jointData']['limit']
    if part_data['joint'] == 'hinge':     # ======= only rotate. =========
        if limit['noLimit']:
            return [0, 0, -np.pi, np.pi]
        else:
            return [0, 0, degree2rad(limit['a']), degree2rad(limit['b'])]
    elif part_data['joint'] == 'slider':
        if not limit.get('rotates'): # ======= only slide =========
            return [limit['a'], limit['b'], 0, 0]
        else:
            if limit['noRotationLimit']:
                return [limit['a'], limit['b'], -np.pi, np.pi]
            else:                # ======= both =========
                rotate_limit = limit['rotationLimit']
                rotate_limit = degree2rad(rotate_limit)
                return [limit['a'], limit['b'], -rotate_limit, rotate_limit]
    else:
        raise NotImplementedError

dfn = 0
id_to_dfn = {}
def calcuate_dfn(parts, cur_id):
    global dfn
    child = list(filter(lambda x: x['raw_parent'] == cur_id, parts))
    child.sort(key=lambda x: x['name'])
    dfn += 1
    id_to_dfn[cur_id] = dfn
    for c in child:
        calcuate_dfn(parts, c['raw_id'])

def process(shape_path:Path, output_info_path:Path, output_mesh_path:Path, needed_categories:list[str]):
    start_time      = time.time()
    raw_meta_path   = Path(shape_path) / 'meta.json'
    raw_meta        = json.loads(raw_meta_path.read_text())
    catecory_name   = raw_meta['model_cat']
    meta            = {'catecory': raw_meta['model_cat'], 'shape_id': raw_meta['anno_id'], 'shape_path': shape_path.as_posix()}
    key_name        = f"{catecory_name}_{meta['shape_id']}"
    processed_part  = []

    path_shape_id   = shape_path.stem

    mobility_file = json.loads((Path(shape_path) / 'mobility_v2.json').read_text())

    if catecory_name not in needed_categories and '*' not in needed_categories:
        return f"[Skip] {catecory_name} is not in needed categories.", shape_path
    print('Processing:', shape_path)

    partid_to_objs = parse_partid_to_objs(shape_path)
    # print('parse_partid_to_objs', partid_to_objs)

    for part in mobility_file:
        new_part = {}
        pid = part['id']

        new_part['raw_id'] = pid
        new_part['raw_parent'] = part['parent']
        new_part['name'] = part['name'].replace('_', ' ')

        # Get Meshs in `obj`
        partids_in_result = [obj['id'] for obj in part["parts"]]
        objs_file = set()
        for partid_in_result in partids_in_result:
            objs_file |= partid_to_objs[partid_in_result]

        meshs_paths = [shape_path / 'textured_objs' / (obj + '.obj')
                         for obj in objs_file]
        merged_mesh_name = f"{catecory_name}_{meta['shape_id']}_{pid}.ply"
        merged_mesh_save_path = output_mesh_path / merged_mesh_name
        mesh = merge_meshs(meshs_paths, merged_mesh_save_path)

        # obb_merged_mesh_name = merged_mesh_name # f"{catecory_name}_{meta['shape_id']}_{pid}_obb.ply"
        # obb_merged_mesh_save_path = output_mesh_path / obb_merged_mesh_name

        new_part['mesh'] = merged_mesh_name

        # part mesh: mesh.
        # obb_parameters = get_oriented_bounding_box_parameters_and_save(merged_mesh_save_path, obb_merged_mesh_save_path)
        # new_part['obbx'] = obb_parameters

        bounding_box = mesh.bounds
        new_part['bbx'] = [bounding_box[0::2], bounding_box[1::2]]

        if part['parent'] != -1:
            new_part['joint_data_origin'] = part['jointData']['axis']['origin']
            new_part['joint_data_direction'] = part['jointData']['axis']['direction']
            if None in new_part['joint_data_direction']:
                return f"[Error]: Bad data in {shape_path.as_posix()}: {new_part['joint_data_direction']}", key_name
        else:
            new_part['joint_data_origin'] = [0, 0, 0]
            new_part['joint_data_direction'] = [0, 0, 0]
        new_part['limit'] = parse_limit(part)

        processed_part.append(new_part)

    global dfn
    global id_to_dfn
    dfn = 0
    id_to_dfn = {-1: 0}
    root_cnt = 0
    for part in processed_part:
        if part['raw_parent'] == -1:
            calcuate_dfn(processed_part, part['raw_id'])
            root_cnt += 1

    if root_cnt > 1:
        return "[Error]: More than one root part.", key_name

    for part in processed_part:
        part['dfn'] = id_to_dfn[part['raw_id']]
        part['dfn_fa'] = id_to_dfn[part['raw_parent']]

    for part in processed_part:
        part.pop('raw_id')
        part.pop('raw_parent')

    output_info_path = output_info_path / (key_name + ".json")
    output_info_path.write_text(json.dumps({
            'meta': meta,
            'part': processed_part
        } , cls=HighPrecisionJsonEncoder, indent=2))

    end_time = time.time()
    return f'[Done] {shape_path.as_posix()} time: {end_time - start_time:.2f}s', key_name


if __name__ == '__main__':
    raw_dataset_paths   = glob('../datasets/0_raw_dataset/*')
    output_info_path    = Path('../datasets/1_preprocessed_info')
    output_mesh_path    = Path('../datasets/1_preprocessed_mesh')
    train_split_ratio   = 1
    needed_categories   = [
            'Bottle',
            'USB',
            'StorageFurniture',
            'Safe'
        ]

    meta_info    = Path('../datasets/meta.json') # Save Split.

    shutil.rmtree(output_info_path, ignore_errors=True)
    shutil.rmtree(output_mesh_path, ignore_errors=True)

    output_info_path.mkdir(exist_ok=True)
    output_mesh_path.mkdir(exist_ok=True)

    failed_shape_path = {}
    success_shape_path = {}
    category_count = {k : 0 for k in needed_categories}

    with Pool(3) as p:
        results = [
            p.apply_async(process, (Path(shape_path), output_info_path, output_mesh_path, needed_categories))
            for shape_path in tqdm(raw_dataset_paths)
        ]

        bar = tqdm(total=len(raw_dataset_paths), desc='Converting meshes')
        while results:
            for r in results:
                if not r.ready(): continue

                bar.update(1)
                status = r.get()

                if 'Error' in status[0]:
                    failed_shape_path[status[1]] =  status[0]
                elif 'Done' in status[0]:
                    success_shape_path[status[1]] =  status[0]
                    category = status[1].split('_')[0]
                    category_count[category] += 1
                if 'Skip' not in status[0]:
                    print(status)
                results.remove(r)

            time.sleep(0.1)

    success_shape_key_name = list(success_shape_path.keys())

    random.shuffle(success_shape_key_name)

    train_split = success_shape_key_name[:int(len(success_shape_key_name) * train_split_ratio)]
    test_split = success_shape_key_name[int(len(success_shape_key_name) * train_split_ratio):]

    print('Failed shape path:', failed_shape_path)
    print('# Failed shape:', len(failed_shape_path))
    print('# Success shape:', len(success_shape_path))
    print('Done category count: ', category_count)

    with open(meta_info, 'w') as f:
        json.dump({"1_extract_from_raw_dataset": {
            'created_date_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'train_split': train_split,
            'test_split': test_split,
            'train_split_count': len(train_split),
            'test_split_count': len(test_split),
            'train_split_ratio': len(train_split) / (len(train_split) + len(test_split)),
            'selected_categories': needed_categories,
            'selected_categories_size': len(failed_shape_path) + len(success_shape_path),
            'failed_shape_count': len(failed_shape_path),
            'failed_ratio': len(failed_shape_path) / (len(failed_shape_path) + len(success_shape_path)),
            'failed_shape_path':  failed_shape_path,
            'success_shape_path': success_shape_path,
            'category_count': category_count
        }}, f, indent=2)



    # print(tt_list)