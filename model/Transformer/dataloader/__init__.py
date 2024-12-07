import torch
import json
import random
import numpy as np
from tqdm import trange
import copy
from glob import glob
from pathlib import Path
from torch.utils.data import dataset

class TransDiffusionDataset(dataset.Dataset):
    def __init__(self, dataset_path: str, cut_off: int, enc_data_fieldname: str, cache_data: bool=True):
        self.dataset_root_path = Path(dataset_path)

        assert enc_data_fieldname in ['description', 'images']
        self.enc_data_fieldname = enc_data_fieldname

        # import meta.json
        self.meta = json.loads((self.dataset_root_path / 'meta.json').read_text())
        self.max_count_token = self.meta['max_count_token']
        self.start_token = torch.tensor(self.meta['start_token'], dtype=torch.float32)
        self.end_token = torch.tensor(self.meta['end_token'], dtype=torch.float32)
        self.pad_token = torch.tensor(self.meta['pad_token'], dtype=torch.float32)

        # get all json files
        all_json_files = self.dataset_root_path.glob('*.json')
        all_json_files = list(filter(lambda x: 'meta.json' not in str(x), all_json_files))
        # if self.enc_data_fieldname == 'description':
        self.files_path = [
                (Path('data') / desc_path, file)
                for file in all_json_files
                for desc_path in json.loads(file.read_text())[self.enc_data_fieldname]
            ]
        # else:
        #     self.files_path = [
        #             (Path('data') / json.loads(file.read_text())['images'], file)
        #             for file in all_json_files
        #         ]


        random.seed(0)
        random.shuffle(self.files_path)

        if cut_off > 0:
            self.files_path = self.files_path[:cut_off]

        self.cut_off = cut_off

        self.cache = [None] * self.__len__()
        if cache_data:
            for i in trange(len(self.cache), desc="caching data"):
                self.cache[i] = self.__getitem__(i)

    def get_best_diffusion_ckpt_path(self):
        return self.meta['best_diffusion_ckpt_path']

    # def get_best_sdf_ckpt_path(self):
    #     return self.meta['best_sdf_ckpt_path']

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        if self.cache[index] is not None:
            return self.cache[index]

        # print(f"Loading {index}th data")
        enc_path, file_path = self.files_path[index]
        data = json.loads(Path(file_path).read_text())

        # print(f"enc_path: {enc_path}")
        enc = np.load(enc_path, allow_pickle=True)

        if self.enc_data_fieldname == 'description':
            enc = enc.item()

        total_token = len(data['exist_node'])
        assert len(data['exist_node']) == len(data['inferenced_token'])

        # Process Input
        input = data['exist_node']
        # with open('input.json', 'w') as f:
        #     json.dump(input, f, indent=4)

        for node_idx, node in enumerate(input):
            raw_data_info = node['token'][:16]
            assert len(node['token'][16:]) == 768
            text_hat = node['packed_info']['text_hat']

            node['token'] = torch.tensor(raw_data_info + text_hat, dtype=torch.float32)
            dfn_fa = node['dfn_fa']
            for idx in range(len(input)):
                if input[idx]['dfn'] == dfn_fa:
                    node['fa'] = idx
                    break
            assert 'fa' in node, f"Can't find father node for {node['dfn']}"
            assert node['fa'] <= node_idx

        for node in input:
            if node.get('dfn') is not None: del node['dfn']
            if node.get('dfn_fa') is not None: del node['dfn_fa']

        for _ in range(self.max_count_token - len(input)):
            input.append({'token': copy.deepcopy(self.pad_token[:80]), 'fa': 0})

        transformed_input = {
                'token': torch.stack([node['token'] for node in input]),
                'fa': torch.tensor([node['fa'] for node in input], dtype=torch.int)
            }

        # Process Output
        infer_nodes = data['inferenced_token']
        output = []
        # 1:   not end token,    0: end token
        output_skip_end_token_mask = []
        for node in infer_nodes:
            node['packed_info'] = {
                'z_logits': torch.tensor(node['packed_info']['z_logits']),
                'latent': torch.tensor(node['packed_info']['latent']),
                'text_hat': torch.tensor(node['packed_info']['text_hat'])
            }
            output.append({
                'token': torch.tensor(node['token'], dtype=torch.float32),
                'packed_info': node['packed_info']
            })
            output_skip_end_token_mask.append(0 if node['dfn'] == -1 else 1)

        for _ in range(self.max_count_token - len(output)):
            output.append({
                'token': copy.deepcopy(self.pad_token),
                'packed_info': node['packed_info'] # node here is not impertant. It is padding token, just for batching data.
            })
            output_skip_end_token_mask.append(1)


        # (seq, attribute) --> (attribute, seq)
        transformed_output = {
                'token': torch.stack([node['token'] for node in output]),
                'packed_info': {
                    'z_logits': torch.stack([node['packed_info']['z_logits'] for node in output]),
                    'latent': torch.stack([node['packed_info']['latent'] for node in output]),
                    'text_hat': torch.stack([node['packed_info']['text_hat'] for node in output])
                }
            }

        output_skip_end_token_mask = torch.tensor(output_skip_end_token_mask, dtype=torch.int)

        # Process Padding Mask
        padding_mask = torch.ones(self.max_count_token, dtype=torch.int16)
        padding_mask[total_token:] = 0

        return [transformed_input, transformed_output, padding_mask, output_skip_end_token_mask] +   \
                    ([enc['encoded_text'], enc['text']] if self.enc_data_fieldname == 'description'
                else [enc.astype(np.float32), str(enc_path)])