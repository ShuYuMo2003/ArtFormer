import json
from tqdm import trange
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class DiffusionDataset(Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()
        dataset_path = Path(dataset_path)
        self.text_latentcode_dir = list(dataset_path.glob('*.npz'))

        self.meta = json.loads((dataset_path / "meta.json").read_text())

        self.data = [None] * len(self.text_latentcode_dir)
        for idx in trange(len(self.text_latentcode_dir), desc="Loading and cache data."):
            data = np.load(self.text_latentcode_dir[idx], allow_pickle=True)
            self.data[idx] = data['text'].astype(np.float32),           \
                             data['latent_code'].astype(np.float32) # ,    \
                            #  data['bounding_box'].astype(np.float32)

    def get_gensdf_ckpt_path(self):
        return self.meta['ckpt']

    def __len__(self):
        return len(self.text_latentcode_dir)

    def __getitem__(self, index):
        return self.data[index]