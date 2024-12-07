import torch, numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

import argparse
import pickle, time, json
import os, os.path as osp

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

from pytorch3d.loss import chamfer_distance
# from tqdm import tqdm
from utils import get_trans_matrix, apply_transformations, sample_object, remove_useless_keys, sample_pcl_for_each_part

def sample_random_pose(obj: list, n_sample: int) -> list:
    """
    uniformly sample a pose for the input object.
    obj: list of parts, each part is a dict containing ['points', 'joint_data_origin', 'joint_data_direction', 'limit']
    returns: object list with a new key 'poses' added to each part, 
            which is a list of 4x4 SE(3) transformation matrix,
            the length of the list is n_sample.
    """
    for part in obj:
        ratios = np.random.uniform(0, 1, size=n_sample)
        part['poses'] = [get_trans_matrix(part, ratio) for ratio in ratios]
    # caclulate the global transformation matrix, with the first part as the root
    P_dict, fa = {}, {}
    for part in obj:
        part_id = part['dfn']
        fa_id = part['dfn_fa']
        P_dict[part_id] = part['poses']
        fa[part_id] = fa_id
    id_list = sorted(list(P_dict.keys()))
    for node_id in id_list:
        if node_id != 0 and fa[node_id] != -1 and fa[node_id] in P_dict:
            for i in range(n_sample):
                P_dict[node_id][i] = P_dict[fa[node_id]][i] @ P_dict[node_id][i]
    return

def fix_pose(obj, N_states, N_pcl, device):
    # apply transformation and merge all parts, result shape: [N_states, N_pcl, 3]
    x_list = []
    for state_id in range(N_states):
        x = []
        for part in obj:
            pp = torch.tensor(part['points'][:, :3], device=device)
            x.append(apply_transformations(pp, part['poses'][state_id]))
        x = torch.cat(x, dim=0).to(device) # [len(obj)*N_pcl, 3]
        # assert x.shape[0] >= N_pcl, f"object has less points {x.shape[0]} than N_pcl={N_pcl}"
        # randomly sample N_pcl points
        idx = torch.randperm(x.shape[0])[:N_pcl]
        x_list.append(x[idx])
    x_list = torch.stack(x_list, dim=0)  # N_states, N_pcl, 3
    ###########
    return x_list

@torch.no_grad()
def compute_instantiation_distance_pair(
    x1_list, x2_list, device=torch.device("cuda:0"), N_states=10, N_pcl=2048
):
    # compute N_states x N_states distance matrix
    # each row is a A, each col is a B
    D = torch.zeros(N_states, N_states, device=device)
    for i in range(N_states):
        cd, _ = chamfer_distance(x1_list[i:i+1].expand(N_states, -1, -1), x2_list, batch_reduction=None)
        D[i] = cd
    # D: [N_states, N_states]
    # D_ref = torch.zeros(N_states, N_states, device=device)
    # for i in range(N_states):
    #     for j in range(N_states):
    #         cd, _ = chamfer_distance(x1_list[i:i+1], x2_list[j:j+1])
    #         D_ref[i, j] = cd
    ###########

    dl, dr = D.min(dim=1).values, D.min(dim=0).values
    dl, dr = dl.mean(), dr.mean()
    distance = dl + dr  # ! note, it's sum

    # gather them in correct way
    return float(distance.cpu().numpy())

def compute_D_matrix(rank, world_size, in_dir, gt_dir, save_dir, N_states=10, N_pcl=2048, sample_file:dict=None):
    """
    compute the D matrix: instantiation distance between each pair of (in_obj, gt)
    in_dir: str, path to the directory containing the object data, each object should be a single .dat file.
    gt_dir: str, path to the directory containing the ground truth object data, each object should be a single .dat file.
    save_dir: str, path to the directory to save the .npz output, which contains [D, gen_fn_list, gt_fn_list, N_states, N_pcl_max, gen_dir, gt_dir].
    """
    print(f"worker {rank} started ...")
    setup(rank, world_size)
    
    # prepare files
    in_fn_list = sorted([f for f in os.listdir(in_dir) if f.endswith(".dat")])
    gt_fn_list = sorted([f for f in os.listdir(gt_dir) if f.endswith(".dat")])
    if sample_file is not None:
        sample_object(in_fn_list, sample_file)
        sample_object(gt_fn_list, sample_file)
        if len(in_fn_list) != len(gt_fn_list):
            if len(in_fn_list) > len(gt_fn_list):
                raise NotImplementedError(f"in_fn_list > gt_fn_list: {len(in_fn_list)} > {len(gt_fn_list)}")
            else:
                sample_object(gt_fn_list, {"fn_list": in_fn_list})
        print(f"{rank}: sampled {len(in_fn_list)} in-objects, {len(gt_fn_list)} ground truth objects.")
    N_in, N_gt = len(in_fn_list), len(gt_fn_list)

    gen_name = osp.basename(in_dir)
    ref_name = osp.basename(gt_dir)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    ###########

    # cache DATA
    DATA_IN, DATA_GT = [], []
    print(f"{rank}: caching input ...")
    for i in range(N_in):
        fn = osp.join(in_dir, in_fn_list[i])
        data = pickle.load(open(fn, "rb"))
        remove_useless_keys(data)
        sample_pcl_for_each_part(data, N_pcl)
        DATA_IN.append(data)
    print(f"{rank}: caching ground truth ...")
    for i in range(N_gt):
        fn = osp.join(gt_dir, gt_fn_list[i])
        data = pickle.load(open(fn, "rb"))
        remove_useless_keys(data)
        sample_pcl_for_each_part(data, N_pcl)
        DATA_GT.append(data)
    ###########

    # sample random poses
    print(f"{rank}: sampling random poses ...")
    for i in range(N_in):
        sample_random_pose(DATA_IN[i], N_states)
    for i in range(N_gt):
        sample_random_pose(DATA_GT[i], N_states)
    
    for i in range(N_in):
        DATA_IN[i] = fix_pose(DATA_IN[i], N_states, N_pcl, torch.device(f"cuda:{rank}"))
    for i in range(N_gt):
        DATA_GT[i] = fix_pose(DATA_GT[i], N_states, N_pcl, torch.device(f"cuda:{rank}"))
    ###########

    # compute D matrix
    requests = []
    print(f"{rank}: computing D matrix ...")
    for i in range(N_in):
        for j in range(N_gt):
            obj_in, obj_gt = DATA_IN[i], DATA_GT[j]
            requests.append((
                (i, j, 0),
                obj_in,
                obj_gt,
                N_states,
                N_pcl,
            ))
    for i in range(N_in):
        for j in range(i + 1, N_in):
            obj_in, obj_gt = DATA_IN[i], DATA_IN[j]
            requests.append((
                (i, j, 1),
                obj_in,
                obj_gt,
                N_states,
                N_pcl,
            ))
    for i in range(N_gt):
        for j in range(i + 1, N_gt):
            obj_in, obj_gt = DATA_GT[i], DATA_GT[j]
            requests.append((
                (i, j, 2),
                obj_in,
                obj_gt,
                N_states,
                N_pcl,
            ))
    ###########

    # parallel computation
    D0 = torch.zeros(N_in, N_gt, device=torch.device(f"cuda:{rank}"))
    D1 = torch.zeros(N_in, N_in, device=torch.device(f"cuda:{rank}"))
    D2 = torch.zeros(N_gt, N_gt, device=torch.device(f"cuda:{rank}"))
    for index, request in enumerate(requests):
        if index % world_size != rank:
            continue
        (_i, _j, _g), obj_in, obj_gt, N_states, N_pcl = request
        _d = compute_instantiation_distance_pair(
            obj_in,
            obj_gt,
            N_states=N_states,
            N_pcl=N_pcl,
            device=torch.device(f"cuda:{rank}"),
        )
        if _g == 0:
            D0[_i, _j] = _d
        elif _g == 1:
            D1[_i, _j] = D1[_j, _i] = _d
        elif _g == 2:
            D2[_i, _j] = D2[_j, _i] = _d
    ###########

    # gather D matrix
    dist.barrier()
    dist.all_reduce(D0, op=dist.ReduceOp.SUM)
    dist.all_reduce(D1, op=dist.ReduceOp.SUM)
    dist.all_reduce(D2, op=dist.ReduceOp.SUM)
    ###########

    # save D matrix to file
    if rank == 0:
        save_name1 = f"{gen_name}_{ref_name}_{N_states}_{N_pcl}.npz"
        save_fn1 = osp.join(save_dir, save_name1)
        D0 = D0.cpu().numpy()
        np.savez_compressed(
            save_fn1,
            D=D0,
            gen_fn_list=in_fn_list,
            gt_fn_list=gt_fn_list,
            N_states=N_states,
            N_pcl=N_pcl,
            gen_dir=in_dir,
            gt_dir=gt_dir,
        )
        print(f"saved to {save_fn1}")

        save_name2 = f"{gen_name}_{gen_name}_{N_states}_{N_pcl}.npz"
        save_fn2 = osp.join(save_dir, save_name2)
        D1 = D1.cpu().numpy()
        np.savez_compressed(
            save_fn2,
            D=D1,
            gen_fn_list=in_fn_list,
            gt_fn_list=in_fn_list,
            N_states=N_states,
            N_pcl=N_pcl,
            gen_dir=in_dir,
            gt_dir=in_dir,
        )
        print(f"saved to {save_fn2}")

        save_name3 = f"{ref_name}_{ref_name}_{N_states}_{N_pcl}.npz"
        save_fn3 = osp.join(save_dir, save_name3)
        D2 = D2.cpu().numpy()
        np.savez_compressed(
            save_fn3,
            D=D2,
            gen_fn_list=gt_fn_list,
            gt_fn_list=gt_fn_list,
            N_states=N_states,
            N_pcl=N_pcl,
            gen_dir=gt_dir,
            gt_dir=gt_dir,
        )
        print(f"saved to {save_fn3}")
    
    cleanup()
    return

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--gt_dir', type=str, required=True,
                            help='Path to the directory containing the ground truth object data, each object should be a single .dat file.')
    arg_parser.add_argument('--data_dir', type=str, required=True,
                            help='Path to the directory containing the object data, each object should be a single .dat file.')
    arg_parser.add_argument('--output_dir', type=str, required=True,
                            help='Path to the directory to save the output.\nOutput will be a .npz file containing the D matrix, which is the instantiation distance between each pair of (in_obj, gt).')
    arg_parser.add_argument('--N_states', type=int, default=10,
                            help='Number of states to sample for each object.')
    arg_parser.add_argument('--N_pcl', type=int, default=2048,
                            help='Number of points to sample for each object.')
    arg_parser.add_argument('--world_size', type=int, default=1,
                            help='Number of processes to use.')
    arg_parser.add_argument('--sample_file_path', type=str, default=None,
                            help='Path to the sample file, which contains the list of object filenames to process.')
    args = arg_parser.parse_args()

    with open(args.sample_file_path, "r") as f:
        sample_file = json.load(f)

    start = time.time()
    mp.spawn(
        compute_D_matrix,
        args=(args.world_size, args.data_dir, args.gt_dir, args.output_dir, args.N_states, args.N_pcl, sample_file),
        nprocs=args.world_size
    )
    print(f"elapsed time: {time.time() - start:.2f} s")

    #################
    # (40 objects) elapsed time: 159.94 s