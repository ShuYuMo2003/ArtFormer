import torch, numpy as np
import argparse
import pickle, time
import os, os.path as osp
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from utils import get_trans_matrix, apply_transformations

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

@torch.no_grad()
def compute_instantiation_distance_pair(
    objA, objB, device=torch.device("cuda:0"), N_states=10, N_pcl=2048
):
    # apply transformation and merge all parts, result shape: [N_states, N_pcl, 3]
    x1_list, x2_list = [], []
    for state_id in range(N_states):
        x1, x2 = [], []
        for partA, partB in zip(objA, objB):
            pA = torch.tensor(partA['points'][:, :3], device=device)
            pB = torch.tensor(partB['points'][:, :3], device=device)
            x1.append(apply_transformations(pA, partA['poses'][state_id]))
            x2.append(apply_transformations(pB, partB['poses'][state_id]))
        x1 = torch.cat(x1, dim=0).to(device) # [N_pcl, 3]
        x2 = torch.cat(x2, dim=0).to(device) # [N_pcl, 3]
        assert x1.shape[0] >= N_pcl, f"object has less points {x1.shape[0]} than N_pcl={N_pcl}"
        assert x2.shape[0] >= N_pcl, f"object has less points {x2.shape[0]} than N_pcl={N_pcl}"
        # randomly sample N_pcl points
        idx1 = torch.randperm(x1.shape[0])[:N_pcl]
        idx2 = torch.randperm(x2.shape[0])[:N_pcl]
        x1_list.append(x1[idx1])
        x2_list.append(x2[idx2])
    x1_list = torch.stack(x1_list, dim=0)  # N_states, N_pcl, 3
    x2_list = torch.stack(x2_list, dim=0)  # N_states, N_pcl, 3
    ###########

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

def compute_D_matrix(in_dir, gt_dir, save_dir, N_states=10, N_pcl=2048):
    """
    compute the D matrix: instantiation distance between each pair of (in_obj, gt)
    in_dir: str, path to the directory containing the object data, each object should be a single .dat file.
    gt_dir: str, path to the directory containing the ground truth object data, each object should be a single .dat file.
    save_dir: str, path to the directory to save the .npz output, which contains [D, gen_fn_list, gt_fn_list, N_states, N_pcl_max, gen_dir, gt_dir].
    """
    # prepare files
    in_fn_list = sorted([f for f in os.listdir(in_dir) if f.endswith(".dat")])
    gt_fn_list = sorted([f for f in os.listdir(gt_dir) if f.endswith(".dat")])
    N_in, N_gt = len(in_fn_list), len(gt_fn_list)
    D = -1.0 * np.ones((N_in, N_gt), dtype=np.float32)

    gen_name = osp.basename(in_dir)
    ref_name = osp.basename(gt_dir)
    save_name = f"{gen_name}_{ref_name}_{N_states}_{N_pcl}.npz"
    save_fn = osp.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    ###########

    # cache DATA
    sym_flag = in_dir == gt_dir
    DATA_IN, DATA_GT = [], []
    print("caching INPUT ...")
    for i in tqdm(range(N_in)):
        fn = osp.join(in_dir, in_fn_list[i])
        data = pickle.load(open(fn, "rb"))
        DATA_IN.append(data)
    if sym_flag:
        DATA_GT = DATA_IN
    else:
        print("caching REF ...")
        for i in tqdm(range(N_gt)):
            fn = osp.join(gt_dir, gt_fn_list[i])
            data = pickle.load(open(fn, "rb"))
            DATA_GT.append(data)
    ###########

    # sample random poses
    print("sampling random poses ...")
    for i in tqdm(range(N_in)):
        sample_random_pose(DATA_IN[i], N_states)
    if not sym_flag:
        for i in tqdm(range(N_gt)):
            sample_random_pose(DATA_GT[i], N_states)
    ###########

    # compute D matrix
    print("computing D matrix ...")
    for i in tqdm(range(N_in)):
        for j in tqdm(range(N_gt)):
            if sym_flag and i == j:
                D[i, j] = 0.0
                continue
            if sym_flag and i > j:
                assert D[j, i] >= 0.0
                D[i, j] = D[j, i]
                continue
            obj_in, obj_gt = DATA_IN[i], DATA_GT[j]
            _d = compute_instantiation_distance_pair(
                obj_in,
                obj_gt,
                N_states=N_states,
                N_pcl=N_pcl,
            )
            D[i, j] = _d
    ###########
    
    # save D matrix to file
    assert (D >= 0.0).all(), "invalid D matrix, some values are negative"
    np.savez_compressed(
        save_fn,
        D=D,
        gen_fn_list=in_fn_list,
        gt_fn_list=gt_fn_list,
        N_states=N_states,
        N_pcl=N_pcl,
        gen_dir=in_dir,
        gt_dir=gt_dir,
    )
    print(f"saved to {save_fn}")
    return D

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
    args = arg_parser.parse_args()

    start = time.time()
    compute_D_matrix(args.data_dir, args.gt_dir, args.output_dir, args.N_states, args.N_pcl)
    compute_D_matrix(args.data_dir, args.data_dir, args.output_dir, args.N_states, args.N_pcl)
    compute_D_matrix(args.gt_dir, args.gt_dir, args.output_dir, args.N_states, args.N_pcl)
    print(f"elapsed time: {time.time()-start:.2f} s")