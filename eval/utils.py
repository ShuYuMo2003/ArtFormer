import torch
import numpy as np

from tqdm import tqdm
from rich import print
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ten_def = {
    "device": device,
    "dtype": torch.float64
}

def calc_linear_value(L, R, ratio):
    return 1.0 * L + (R - L) * 1.0 * ratio

def produce_rotate_matrix(direction, angle):
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)

    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, **ten_def)

    direction = direction / direction.norm(p=2)
    K = torch.tensor([
        [0, -direction[2], direction[1]],
        [direction[2], 0, -direction[0]],
        [-direction[1], direction[0], 0]
    ], **ten_def)
    R = torch.eye(3, **ten_def) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    M = torch.eye(4, **ten_def)
    M[0:3, 0:3] = R
    return M

def produce_translate_matrix(direction, distance):
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)
    M = torch.eye(4, **ten_def)
    M[0:3, 3] = direction * distance
    return M

def produce_rotate_around_line_matrix(start, direction, angle):
    if not torch.is_tensor(start):
        start = torch.tensor(start, **ten_def)
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)
    T = produce_translate_matrix(-start, 1)
    R = produce_rotate_matrix(direction, angle)
    T_inv = produce_translate_matrix(start, 1)
    return T_inv @ R @ T

def get_trans_matrix(part_dict, ratio):
    """
    calcutate 4*4 SE(3) transformation matrix
    ratio: float within [0, 1.0], corresponding to the translation ratio
    """
    distance = calc_linear_value(*part_dict['limit'][:2], ratio)
    Mt = produce_translate_matrix(part_dict['joint_data_direction'], distance)

    angle = calc_linear_value(*part_dict['limit'][2:], ratio)
    Mr = produce_rotate_around_line_matrix(part_dict['joint_data_origin'], part_dict['joint_data_direction'], angle)

    M = Mr @ Mt
    return M

def apply_transformations(points, M):
    """
    apply the transformation matrix to the points
    points: n*3 array, points to be transformed
    M: 4*4 array, transformation matrix
    """
    points = torch.cat((points, torch.ones((points.shape[0], 1), **ten_def)), dim=1)
    transformed = (M @ points.T).T
    return transformed[:, :3]


#################################################
# utils for computing metrics based on D matrix #
#################################################

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }

# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    # print(pred)
    # print(label)
    return s

def eval_ID(root_dir, gen_name, ref_name, N_states=10, N_pcl=2048):
    results = {}
    rs_fn = f"{root_dir}/{gen_name}_{ref_name}_{N_states}_{N_pcl}.npz"
    rr_fn = f"{root_dir}/{ref_name}_{ref_name}_{N_states}_{N_pcl}.npz"
    ss_fn = f"{root_dir}/{gen_name}_{gen_name}_{N_states}_{N_pcl}.npz"
    M_rs = torch.from_numpy(np.load(rs_fn)["D"])
    M_rr = torch.from_numpy(np.load(rr_fn)["D"])
    M_ss = torch.from_numpy(np.load(ss_fn)["D"])
    print(M_rs[:5,:5])
    ret = lgan_mmd_cov(M_rs.t())
    results.update({
        "%s-ID" % k: v for k, v in ret.items()
    })
    ret = knn(M_rr, M_rs, M_ss, 1, sqrt=False)
    results.update({
        "1-NN-ID-%s" % k: v for k, v in ret.items() if 'acc' in k
    })
    # print(M_rs[:5,:5])
    print(gen_name, ref_name)
    # pprint(results)
    final_results = {
        "1-NN-ID-acc": results["1-NN-ID-acc"],
        "lgan_mmd-ID": results["lgan_mmd-ID"],
        "lgam_cov-ID": results["lgan_cov-ID"],
    }
    # final_results = {
    #     "1-NN-ID-acc": float(results["1-NN-ID-acc"]),
    #     "lgan_mmd-ID": float(results["lgan_mmd-ID"]),
    #     "lgam_cov-ID": float(results["lgan_cov-ID"]),
    # }
    pprint(final_results)
    return



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ten_def = {
    "device": device,
    "dtype": torch.float64
}

def calc_linear_value(L, R, ratio):
    return 1.0 * L + (R - L) * 1.0 * ratio

def produce_rotate_matrix(direction, angle):
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)

    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, **ten_def)

    direction = direction / direction.norm(p=2)
    K = torch.tensor([
        [0, -direction[2], direction[1]],
        [direction[2], 0, -direction[0]],
        [-direction[1], direction[0], 0]
    ], **ten_def)
    R = torch.eye(3, **ten_def) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    M = torch.eye(4, **ten_def)
    M[0:3, 0:3] = R
    return M

def produce_translate_matrix(direction, distance):
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)
    M = torch.eye(4, **ten_def)
    M[0:3, 3] = direction * distance
    return M

def produce_rotate_around_line_matrix(start, direction, angle):
    if not torch.is_tensor(start):
        start = torch.tensor(start, **ten_def)
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction, **ten_def)
    T = produce_translate_matrix(-start, 1)
    R = produce_rotate_matrix(direction, angle)
    T_inv = produce_translate_matrix(start, 1)
    return T_inv @ R @ T

def get_trans_matrix(part_dict, ratio):
    """
    calcutate 4*4 SE(3) transformation matrix
    ratio: float within [0, 1.0], corresponding to the translation ratio
    """
    # if limit == [0, 0, 0, 0] then the part is fixed
    eps = 1e-6
    if part_dict['limit'][1] - part_dict['limit'][0] <= eps and part_dict['limit'][3] - part_dict['limit'][2] <= eps:
        return torch.eye(4, **ten_def)

    distance = calc_linear_value(*part_dict['limit'][:2], ratio)
    Mt = produce_translate_matrix(part_dict['joint_data_direction'], distance)

    angle = calc_linear_value(*part_dict['limit'][2:], ratio)
    Mr = produce_rotate_around_line_matrix(part_dict['joint_data_origin'], part_dict['joint_data_direction'], angle)

    M = Mr @ Mt
    return M

def prepare_trans_matrix(obj, ratio):
    """
    prepare the transformation matrix for every part in the object
    """

    M_dict, fa = {}, {}

    M_dict[0] = torch.eye(4, **ten_def)

    for part in obj:
        cur_id = part['dfn']
        # print('cur_id = ', cur_id)
        M = get_trans_matrix(part, ratio)
        M_dict[cur_id] = M
        fa[cur_id] = part['dfn_fa']

    keys = list(M_dict.keys())
    keys.sort()
    for cur_id in keys:
        if cur_id != 0 and fa[cur_id] in M_dict:
            M_dict[cur_id] = M_dict[cur_id] @ M_dict[fa[cur_id]]
            assert fa[cur_id] != cur_id
    return M_dict

def apply_transformations(points, M):
    """
    apply the transformation matrix to the points
    points: n*3 array, points to be transformed
    M: 4*4 array, transformation matrix
    """
    points = torch.cat((points, torch.ones((points.shape[0], 1), **ten_def)), dim=1)
    transformed = (M @ points.T).T
    return transformed[:, :3]

def get_ref_seperation(points, n_ref_sample=10):
    """
    assume points to be uniformly distributed, estimate the seperation distance
    """
    n_ref_sample = min(n_ref_sample, points.shape[0] // 2)

    ref_index = torch.randperm(points.shape[0], device=device)
    ref_points = points[ref_index[:n_ref_sample]]
    other_points = points[ref_index[n_ref_sample:]]

    distance = torch.cdist(ref_points, other_points)

    min_distance = distance.min(dim=1).values

    assert min_distance.shape[0] == n_ref_sample

    return min_distance.mean()

def count_intersected_points(query_points, ref_points, sep, n_sample=None):
    """
    count the number of intersected points between query points and reference points
    """
    sample_ratio = 1
    if n_sample is not None:
        n_sample = min(n_sample, query_points.shape[0])
        sample_ratio = n_sample / query_points.shape[0]

        index = torch.randperm(query_points.shape[0], device=device)[:n_sample]
        query_points = query_points[index]
    else:
        n_sample = query_points.shape[0]

    distance = torch.cdist(query_points, ref_points)
    min_distance = distance.min(dim=1).values

    assert min_distance.shape[0] == n_sample

    n_intersected = (min_distance < sep).float().sum()
    return n_intersected / sample_ratio

def sample_iou(trans1, trans2, rho1, rho2, conf_T):
    """
    compute iou between two parts, after applying a SE(3) transformation
    """

    # print('points1.shape = ', str(trans1.shape))
    # print('points2.shape = ', str(trans2.shape))

    # get the seperation distance
    sep1 = get_ref_seperation(trans1)
    # print('sep1 = ', sep1)
    sep2 = get_ref_seperation(trans2)
    # print('sep2 = ', sep2)

    # compute iou
    inter1 = count_intersected_points(trans1, trans2, sep2 * conf_T) / rho1
    # print('inter1 = ', inter1)
    inter2 = count_intersected_points(trans2, trans1, sep1 * conf_T) / rho2
    # print('inter2 = ', inter2)

    inter = (inter1 + inter2) / 2
    union = (trans1.shape[0] / rho1) + (trans2.shape[0] / rho2) - inter
    iou = inter / union
    return iou

def POR(obj, n_sample=None, n_states=10, conf_T=1.5):
    """
    compute the average and maximum Part Overlapping Ratio (POR) of a object
    n_states: number of poses for the object
    conf_T: confidence threshold, the ratio of the seperation distance in [1, 2]
        bigger the value, more false positive (FP)
        smaller the value, more false negative (FN)
    n_sample: if this fucntion is too slow, you can set n_sample to a smaller value (None for no sampling)
    returns: (average POR, maximum POR)
    """
    assert type(obj) == list, """
        obj must be a list of parts with the following structure:
        [
            {
                "points": n*3 np.ndarray,
                "rho": float, part's density, point per unit space(1*1*1)
                "joint_data_origin": [x0, y0, z0],
                "joint_data_direction": [x1, y1, z1],
                "limit": [p_min, p_max, r_min, r_max],
                "dfn": dfs number,
                "dfn_fa": father's dfs number,
            }, ... (other parts)
        ]
        """

    # print("sampling points")
    for part in obj:
        part['points'] = torch.tensor(part['points'], device=device)
        if n_sample is not None:
            index = torch.randperm(part['points'].shape[0], device=device)[:n_sample]
            part['points'] = part['points'][index]

    n_parts = len(obj)
    states = np.linspace(0, 1, n_states)
    results = []
    for state in tqdm(states, desc="Processing on different pose state."):
        M_dict = prepare_trans_matrix(obj, state)

        points = [None] * n_parts
        for i in range(n_parts):
            M = M_dict[obj[i]['dfn']]
            raw_point = obj[i]['points'][:, :3]
            points[i] = apply_transformations(raw_point, M)

        ious = []
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                iou = sample_iou(points[i], points[j], obj[i]['rho'], obj[j]['rho'], conf_T)
                if iou is not None:
                    ious.append(iou)
                # if i == 0 and j == 2:
                #     print(f"State: {state}, Part {i} and Part {j}: {iou}")
        if len(ious) > 0:
            results.extend(ious)

    if len(results) == 0:
        print("warning: no valid iou value is computed.")
        return None, None

    return torch.tensor(results).mean(), torch.tensor(results).max()

def sample_object(fn_list_to_process, sample_file):
    fn_list = sample_file["fn_list"]
    result = []
    for name in fn_list_to_process:
        if name in fn_list:
            result.append(name)
    fn_list_to_process.clear()
    fn_list_to_process.extend(result)
    assert len(fn_list_to_process) <= len(fn_list)
    if len(fn_list_to_process) < len(fn_list):
        print(f"warning: {len(fn_list) - len(fn_list_to_process)} objects are not found in data_dir.")
        for name in fn_list:
            if name not in fn_list_to_process:
                print(name)
    # if len(fn_list_to_process) > len(fn_list):
    #     print(f"warning: {len(fn_list_to_process) - len(fn_list)} objects are not found in sample_file.")

def remove_useless_keys(obj, useless_keys = ['mesh', 'shape_code']):
    for part in obj:
        for key in useless_keys:
            if key in part:
                del part[key]

def align_part_keys(obj):
    for part in obj:
        if 'bbx' in part:
            part['bbox_center'], part['bbox_l'] = part['bbx']
            del part['bbx']

def sample_pcl_for_each_part(obj, N_pcl):
    for part in obj:
        raw_points = part['points']
        if raw_points.shape[0] > N_pcl:
            index = torch.randperm(raw_points.shape[0], device=device)[:N_pcl].tolist()
            part['points'] = raw_points[index]

def repair_dfn(obj):
    cur_dfs_num = 0
    def dfs(cur_part, fa_dfn, obj):
        nonlocal cur_dfs_num
        cur_part['Rdfn'] = cur_dfs_num
        cur_part['Rdfn_fa'] = fa_dfn
        cur_dfs_num += 1

        for part in obj:
            if part['dfn_fa'] == cur_part['dfn']:
                dfs(part, cur_part['Rdfn'], obj)
    
    root = None
    for part in obj:
        if part['dfn_fa'] == -1:
            if root is not None:
                raise ValueError("Multiple root parts detected.")
            root = part
    if root is None:
        raise ValueError("No root part detected.")
    dfs(root, -1, obj)

    for part in obj:
        part['dfn'] = part['Rdfn']
        part['dfn_fa'] = part['Rdfn_fa']
        del part['Rdfn']
        del part['Rdfn_fa']