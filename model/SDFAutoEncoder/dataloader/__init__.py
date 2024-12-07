import json
import random
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
from rich import print

from utils.mylogging import Log
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.base import TransArticulatedBaseDataModule
from utils.mylogging import Log

class GenSDFDataset(Dataset):
    def __init__(self, dataset_dir: Path, train: bool,
                 samples_per_mesh: int, pc_size: int,
                 uniform_sample_ratio: float):
        super().__init__()

        dataset_meta = json.loads((dataset_dir / 'meta.json').read_text())

        assert train is None, "Only suppert train is None."

        # if train is None:
        #     self.current_dataset_keyname = dataset_meta['1_extract_from_raw_dataset']['train_split'] \
        #                                  + dataset_meta['1_extract_from_raw_dataset']['test_split']
        # else:
        #     if train: self.current_dataset_keyname = dataset_meta['1_extract_from_raw_dataset']['train_split']
        #     else:     self.current_dataset_keyname = dataset_meta['1_extract_from_raw_dataset']['test_split']

        self.dataset_dir = list(dataset_dir.glob('2_gensdf_dataset/*.npz'))

        print("Len = ", len(self.dataset_dir))

        # filtered_dataset_dir = []
        # for _file in self.dataset_dir:
        #     file = _file.stem
        #     key_name = '_'.join(file.split('_')[:2])
        #     if key_name in self.current_dataset_keyname:
        #         filtered_dataset_dir.append(_file)
        # self.dataset_dir = filtered_dataset_dir

        random.shuffle(self.dataset_dir)

        self.n_uniform_point = int(samples_per_mesh * uniform_sample_ratio)
        self.n_near_surfcae_point = samples_per_mesh - self.n_uniform_point
        self.pc_size = pc_size

    def __len__(self):
        return len(self.dataset_dir)

    def select_point(self, point, sdf, n_point):
        half = int(n_point / 2)

        neg_idx = np.where(sdf < 0)
        pos_idx = np.where(~(sdf < 0))

        assert len(neg_idx) == 1 and len(pos_idx) == 1

        neg_idx = neg_idx[0]
        pos_idx = pos_idx[0]

        assert neg_idx.shape[0] >= half or pos_idx.shape[0] >= half, 'Not enough points'

        np.random.shuffle(neg_idx)
        np.random.shuffle(pos_idx)

        if neg_idx.shape[0] < half:
            n_point_from_other = half - neg_idx.shape[0]
            neg_idx = np.concatenate((neg_idx, pos_idx[-n_point_from_other:]))
            pos_idx = pos_idx[:-n_point_from_other]

        if pos_idx.shape[0] < half:
            n_point_from_other = half - pos_idx.shape[0]
            pos_idx = np.concatenate((pos_idx, neg_idx[-n_point_from_other:]))
            neg_idx = neg_idx[:-n_point_from_other]

        assert neg_idx.shape[0] >= half and pos_idx.shape[0] >= half, 'Not enough points'

        neg_idx = neg_idx[:half]
        pos_idx = pos_idx[:half]

        idx = np.concatenate([neg_idx, pos_idx])
        np.random.shuffle(idx)

        return point[idx], sdf[idx]

    def __getitem__(self, index):
        file = self.dataset_dir[index]
        data = np.load(file.as_posix(), allow_pickle=True)

        uniform_point, uniform_sdf = self.select_point(data['point_uniform'], data['sdf_uniform'], self.n_uniform_point)
        surface_point, surface_sdf = self.select_point(data['point_surface'], data['sdf_surface'], self.n_near_surfcae_point)

        point_cloud = data['point_on']
        np.random.shuffle(point_cloud)
        point_cloud = point_cloud[:self.pc_size]

        # Convert to float32
        uniform_point   = uniform_point.astype(np.float32)
        uniform_sdf     = uniform_sdf.astype(np.float32)
        surface_point   = surface_point.astype(np.float32)
        surface_sdf     = surface_sdf.astype(np.float32)
        point_cloud     = point_cloud.astype(np.float32)

        xyz = np.concatenate([uniform_point, surface_point])
        gt_sdf = np.concatenate([uniform_sdf, surface_sdf])

        idx = np.random.permutation(xyz.shape[0])

        return {
            'xyz': xyz[idx],
            'gt_sdf': gt_sdf[idx],
            'point_cloud': point_cloud,
            'filename': file.stem
        }
