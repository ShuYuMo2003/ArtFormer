
import os
import json
import shutil
import time
from pathlib import Path
from time import sleep
from rich import print
from rich.console import Console
from rich.table import Column, Table
from multiprocessing import Pool, cpu_count

import sys
sys.path.append('../..')
from utils.mylogging import console, Log

import trimesh
from tqdm import tqdm
import numpy as np


n_sample_point_each = None
uniform_sample_ratio = None
n_point_cloud = None

surface_point_sigma = 0.012
points_padding = 0.05

def create_folder(*paths):
    for path in paths:
        try: os.makedirs(str(path))
        except FileExistsError: pass

def ply_to_obj(ply_file, obj_file):
    mesh = trimesh.load_mesh(
        open(ply_file.as_posix(), 'rb'),
        file_type='ply'
    )
    print('1# bounding box: ', mesh.bounds)
    (min_bound, max_bound) = mesh.bounds
    center = (max_bound + min_bound) / 2
    mesh.vertices -= center

    scale = (max_bound - min_bound).max() / (2 - 0.01)
    mesh.vertices /= scale # fit into [-1, 1]

    print('2# bounding box: ', mesh.bounds)

    mesh.export(obj_file.as_posix(), file_type='obj')

def obj_to_wtobj_by_pcu(obj_file, wt_obj_file):
    import point_cloud_utils as pcu

    v, f = obj_to_wtobj_by_pcu_vf(obj_file)
    pcu.save_mesh_vf(wt_obj_file.as_posix(), v, f)
    return "Done"

def obj_to_wtobj_by_pcu_vf(obj_file):
    import point_cloud_utils as pcu

    # print('read obj file', obj_file)
    v, f = pcu.load_mesh_vf(obj_file)


    # # The resolution parameter controls the density of the output mesh
    # # It is linearly proportional to the number of faces in the output
    # # mesh. A higher value corresponds to a denser mesh.
    resolution = 19_000
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)

    return vw, fw

# @see: https://www.fwilliams.info/point-cloud-utils/sections/mesh_sdf/
def wtobj_to_sdf_by_pcu(wt_obj_file, sdf_file, sample_method=[str, str]):
    '''
        Generate SDF from watertight obj file
        :param wt_obj_file: watertight obj file
        :param sdf_file: output sdf file
        :param sample_method: sample method for 'point near surface' and 'point cloud' respectively
                            choice: 'poisson_disk', 'uniform'
    '''
    import point_cloud_utils as pcu
    _v, _f = pcu.load_mesh_vf(wt_obj_file.as_posix())

    _v = _v.astype(np.float64)

    n_point_total = n_sample_point_each
    n_point_near_surface = int(n_point_total * (1 - uniform_sample_ratio))

    # Generate point near the surface
    Log.info('sampling point near mesh surface using %s', sample_method[0])

    if sample_method[0] == 'poisson_disk':
        fid, bc = pcu.sample_mesh_poisson_disk(_v, _f, num_samples=n_point_near_surface)
    elif sample_method[0] == 'random':
        fid, bc = pcu.sample_mesh_random(_v, _f, num_samples=n_point_near_surface)
    else:
        raise ValueError(f'Invalid method {sample_method[0]}')

    point_near_surface = pcu.interpolate_barycentric_coords(_f, fid, bc, _v)
    n_point_near_surface = point_near_surface.shape[0]
    point_near_surface += surface_point_sigma * np.random.randn(n_point_near_surface, 3)
    Log.info('point_near_surface: %s', point_near_surface.shape)

    # Generate point on the surface (point cloud)
    Log.info('sampling point on mesh surface using %s', sample_method[1])
    n_point_on_surface = n_point_cloud

    if sample_method[1] == 'poisson_disk':
        fid, bc = pcu.sample_mesh_poisson_disk(_v, _f, num_samples=n_point_on_surface)
    elif sample_method[1] == 'random':
        fid, bc = pcu.sample_mesh_random(_v, _f, num_samples=n_point_on_surface)
    else:
        raise ValueError(f'Invalid method {sample_method[1]}')

    point_on_surface = pcu.interpolate_barycentric_coords(_f, fid, bc, _v)
    ## In `interpolate_barycentric_coords()`, the number of points may not be exactly equal to `n_point_on_surface`
    n_point_on_surface = point_on_surface.shape[0]

    # Generate uniform point
    Log.info('sampling point in box')
    n_point_uniform = n_point_total - n_point_near_surface
    box_size = 2 + points_padding
    uniform_point = box_size * np.random.rand(n_point_uniform, 3) - (box_size / 2)

    # Combine surface and uniform point
    query_pts = np.concatenate([point_near_surface, uniform_point, point_on_surface], axis=0)

    Log.info('computing signed distance')
    sdf, fid, bc = pcu.signed_distance_to_mesh(query_pts, _v, _f)

    point_surface   = query_pts[:n_point_near_surface]
    sdf_surface     = sdf[:n_point_near_surface]

    point_uniform   = query_pts[n_point_near_surface:n_point_near_surface+n_point_uniform]
    sdf_uniform     = sdf[n_point_near_surface:n_point_near_surface+n_point_uniform]

    point_on        = query_pts[n_point_near_surface+n_point_uniform:]
    sdf_on          = sdf[n_point_near_surface+n_point_uniform:]

    assert (point_on.shape[0] == n_point_on_surface
        and sdf_on.shape[0] == n_point_on_surface
        and point_uniform.shape[0] == n_point_uniform
        and sdf_uniform.shape[0] == n_point_uniform
        and point_surface.shape[0] == n_point_near_surface
        and sdf_surface.shape[0] == n_point_near_surface), 'Error in point count'

    table = Table(show_header=True, header_style="bold magenta", title=wt_obj_file.stem)
    table.add_column("Item", justify="center")
    table.add_column("Shape", justify="center")
    table.add_column("Occ Rate", justify="center")
    table.add_column("Bounds", justify="center")
    table.add_column("Abs Sdf Range", justify="center")

    table.add_row("Point Uniform", str(point_uniform.shape), f"{(sdf_uniform < 0).astype(np.float32).mean():.4f}",
                  f"{point_uniform.min():.4f} ~ {point_uniform.max():.4f}", f"{np.abs(sdf_uniform).min():.4f} ~ {np.abs(sdf_uniform).max():.4f}")
    table.add_row("Point Surface", str(point_surface.shape), f"{(sdf_surface < 0).astype(np.float32).mean():.4f}",
                  f"{point_surface.min():.4f} ~ {point_surface.max():.4f}", f"{np.abs(sdf_surface).min():.4f} ~ {np.abs(sdf_surface).max():.4f}")
    table.add_row("Point On Mesh", str(point_on.shape), f"{(sdf_on < 0).astype(np.float32).mean():.4f}",
                  f"{point_on.min():.4f} ~ {point_on.max():.4f}", f"{np.abs(sdf_on).min():.4f} ~ {np.abs(sdf_on).max():.4f}")
    table.add_row("Total", str(query_pts.shape), f"{(sdf < 0).astype(np.float32).mean():.4f}",
                  f"{query_pts.min():.4f} ~ {query_pts.max():.4f}", f"{np.abs(sdf).min():.4f} ~ {np.abs(sdf).max():.4f}")

    console.print(table)

    np.savez(sdf_file.as_posix(),
             point_uniform=point_uniform, sdf_uniform=sdf_uniform,
             point_surface=point_surface, sdf_surface=sdf_surface,
             point_on=point_on, sdf_on=sdf_on)

    return "Done"


def convert_mesh(ply_file, clear_temp, wt_method, sdf_method, sample_method=['random', 'random']):
    start_time = time.time()
    stem = ply_file.stem
    temp_dir   = Path(f'../datasets/2_gensdf_dataset/temp/{stem}')
    result_dir = Path(f'../datasets/2_gensdf_dataset/')

    create_folder(temp_dir, result_dir)

    obj_file = temp_dir / (stem + ".obj")
    wt_obj_file = temp_dir / (stem + ".wt.ply")
    sdf_target_file = result_dir / f'{stem}.sdf'

    if (result_dir / f'{stem}.sdf.npz').exists():
        Log.info('Already exists: %s', sdf_target_file)
        return "Done"

    Log.info('(1) Converting to (obj) %s', obj_file)
    ply_to_obj(ply_file, obj_file)

    Log.info('(2) Converting to (wt) %s', wt_obj_file)
    if wt_method == 'pcu':
        Log.info('(2) Using pcu to watertight obj')
        assert obj_to_wtobj_by_pcu(obj_file, wt_obj_file) == 'Done'
    else:
        raise ValueError(f'Invalid method wt_method {wt_method}')

    Log.info('(3) Converting to (sdf) %s', sdf_target_file)
    if sdf_method == 'pcu':
        Log.info('(3) Using pcu to generate sdf')
        if wtobj_to_sdf_by_pcu(wt_obj_file, sdf_target_file, sample_method) != 'Done':
            Log.error('Error in sdf generation')
            return 'Error'
    else:
        raise ValueError(f'Invalid method sdf_method {sdf_method}')

    Log.info(f'finished in {time.time() - start_time:.2f} s')
    if clear_temp: shutil.rmtree(temp_dir, ignore_errors=True)

    return "Done"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate SDF from mesh files.')
    parser.add_argument('--clear_temp_file', type=bool, default=False, help='Clear temp files')

    parser.add_argument('--n_sample_point', type=int, default=400000, help='Number of points for each mesh')
    parser.add_argument('--uniform_sample_ratio', type=float, default=0.5, help='Uniform sample ratio')
    parser.add_argument('--n_point_cloud', type=int, default=100000, help='Number of point cloud')
    parser.add_argument('--n_process', type=int, default=10, help='Number of process')
    parser.add_argument('--near_surface_sammple_method', type=str, default='random', help='Sample method for near surface', choices=['poisson_disk', 'random'])
    parser.add_argument('--on_surface_sample_method', type=str,  default='poisson_disk', help='Sample method for on surface', choices=['poisson_disk', 'random'])

    args = parser.parse_args()

    n_sample_point_each = args.n_sample_point
    uniform_sample_ratio = args.uniform_sample_ratio
    n_point_cloud = args.n_point_cloud
    sample_method = [args.near_surface_sammple_method, args.on_surface_sample_method]

    shutil.rmtree('../datasets/2_gensdf_dataset', ignore_errors=True)
    all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply',
                            Path('../datasets/1_preprocessed_mesh/').iterdir()))

    first_stem_names = list(map(lambda x : x.stem, all_ply_files))


    # whether to use extra mesh.
    # ex_all_ply_files = list(filter(lambda x : x.as_posix()[-3:] == 'ply' and x.stem not in first_stem_names,
    #                         Path('../datasets/1_preprocessed_mesh/ex').iterdir()))

    # all_ply_files.sort(key=lambda x : str(x))
    # ex_all_ply_files.sort(key=lambda x : str(x))\
    ex_all_ply_files = []

    # print(all_ply_files)
    # print(ex_all_ply_files)
    # exit()

    all_ply_files = all_ply_files + ex_all_ply_files

    print("all_ply_files: ", len(all_ply_files))
    print("ex_all_ply_files: ", len(ex_all_ply_files))
    print("expand ratio: ", len(ex_all_ply_files) / (len(all_ply_files) + len(ex_all_ply_files)))

    # all_ply_files = all_ply_files[:5]
    # all_ply_files = [Path('../dataset/1_preprocessed_mesh/USB_64_0.ply')]

    # convert_mesh(all_ply_files[0], args.clear_temp_file, 'pcu', 'pcu', sample_method)
    # exit(0)

    failed = []
    done = []
    with Pool(args.n_process) as p:
        result = [
            (p.apply_async(convert_mesh, (ply_file, args.clear_temp_file, 'pcu', 'pcu', sample_method)), ply_file)
            for ply_file in all_ply_files
        ]
        bar = tqdm(total=len(result), desc='Converting meshes')
        while result:
            for r, ply_file in result:
                if r.ready():
                    bar.update(1)
                    if r.get() != 'Done':
                        Log.error('Error in converting mesh')
                        failed.append(ply_file)
                    else:
                        Log.info('Done: %s', ply_file)
                        done.append(ply_file)
                    result.remove((r, ply_file))
            sleep(0.1)

    meta = json.loads(open('../datasets/meta.json', 'r').read())

    meta['2_generate_gensdf_dataset'] = {
        'create_date_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_sample_point': n_sample_point_each,
        'uniform_sample_ratio': uniform_sample_ratio,
        'n_point_cloud': n_point_cloud,
        'near_surface_sammple_method': sample_method[0],
        'on_surface_sample_method': sample_method[1],
        'all_input_files': [str(f) for f in all_ply_files],
        'all_input_files_count': len(all_ply_files),
        'done': [str(f) for f in done],
        'done_count': len(done),
        'failed': [str(f) for f in failed],
        'failed_count': len(failed),
        'done_ratio': len(done) / len(all_ply_files)
    }

    with open('../datasets/meta.json', 'w') as f:
        f.write(json.dumps(meta, indent=2))

    Log.info('all_ply_files = %s.', len(all_ply_files))
    Log.info('Current Done = %s.', len(done))
    Log.critical('Failed: %s', failed)

