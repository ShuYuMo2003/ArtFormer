import json
import torch
import shutil
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel

import sys
sys.path.append('../..')
from model.SDFAutoEncoder import SDFAutoEncoder
from model.SDFAutoEncoder.dataloader import GenSDFDataset
from utils.mylogging import Log
from utils import to_cuda, camel_to_snake

best_ckpt_path = '<skip, change this after the finish of the training of the SDF model>'

def determine_latentcode_encoder(best_ckpt_path):
    Log.info('Using best ckpt: %s', best_ckpt_path)
    gensdf = SDFAutoEncoder.load_from_checkpoint(best_ckpt_path)
    return gensdf

def evaluate_latent_codes(gensdf):
    dataloader = DataLoader(
            GenSDFDataset(
                    dataset_dir=Path('../datasets'), train=None,
                    samples_per_mesh=16000, pc_size=4096,
                    uniform_sample_ratio=0.3
                ),
            batch_size=20, num_workers=5, pin_memory=True, persistent_workers=True
        )

    print("Length =", len(dataloader.dataset))
    gensdf.eval()
    gensdf = gensdf.to(device)

    path_to_latent = {}
    for batch, batched_data in tqdm(enumerate(dataloader),
                                    desc=f'Evaluating Latent Code', total=len(dataloader)):
        x = to_cuda(batched_data)

        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        with torch.no_grad():
            plane_features = gensdf.encoder.get_plane_features(pc)
            original_features = torch.cat(plane_features, dim=1)
            out = gensdf.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]

        z = out[2]

        for batch in range(z.shape[0]):
            latent = z[batch, ...]
            latent_numpy = latent.detach().cpu().numpy()
            path = x['filename'][batch]
            path = Path(path).stem.replace('.sdf', '') + ".ply"
            path_to_latent[path] = latent_numpy
            # Log.info(f"Latent code for {path} is {latent_numpy.shape}")

    Log.info('Latent code evaluation done. count = %s', len(path_to_latent))
    return path_to_latent

def encode_texts(texts, t5_cache_path, t5_model_name, t5_batch_size, t5_device, t5_max_sentence_length):
    Log.info('Loading T5 model')
    tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix())
    model = T5EncoderModel.from_pretrained(t5_model_name, cache_dir=t5_cache_path.as_posix()).to(device)

    texts = list(texts)
    text_to_e_text = {}
    for s in tqdm(range(0, len(texts), t5_batch_size), desc="Encoding sentences"):
        slice = texts[s:min(s+t5_batch_size, len(texts))]
        # print(slice)
        input_ids = tokenizer(slice, return_tensors="pt", padding='max_length', max_length=t5_max_sentence_length).input_ids
        input_ids = input_ids.to(device)
        outputs = model(input_ids=input_ids)
        encoded_text = outputs.last_hidden_state.detach().cpu().numpy()

        for idx, e_text in enumerate(encoded_text):
            text_to_e_text[slice[idx]] = e_text

    return text_to_e_text

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gensdf_model = determine_latentcode_encoder(best_ckpt_path)

    output_path = Path('../datasets/2.1_text_n_latentcode')
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)
    mesh_info_path = Path('../datasets/1_preprocessed_info')

    t5_cache_path = Path('../../cache/t5_cache')
    t5_cache_path.mkdir(exist_ok=True)
    t5_model_name = 'google-t5/t5-large'
    t5_batch_size = 16
    t5_max_sentence_length = 16

    Log.info('Evaluating latent codes')
    path_to_latent = evaluate_latent_codes(gensdf_model)

    texts = set()

    path_to_bbox_ratio = {}

    for shape_json_path in list(mesh_info_path.glob('*.json')):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = camel_to_snake(shape_json['meta']['catecory'])
        for part_info in shape_json['part']:
            texts.add(f"{shape_name}, {part_info['name']}")

    # ex_mesh_info_path = Path('../datasets/1_preprocessed_info/ex')

    # for ex_shape_json_path in list(ex_mesh_info_path.glob('*.json')):
    #     shape_json = json.loads(ex_shape_json_path.read_text())
    #     shape_name = camel_to_snake(shape_json['meta']['model_cat'])
    #     for part in shape_json['part']:
    #         texts.add(f"{shape_name}, {part['name']}")

    print(texts)

    text_to_e_text = encode_texts(texts, t5_cache_path, t5_model_name, t5_batch_size, device, t5_max_sentence_length)

    failed  = []
    success = []
    for shape_json_path in list(mesh_info_path.glob('*.json')):
        shape_json = json.loads(shape_json_path.read_text())
        shape_name = camel_to_snake(shape_json['meta']['catecory'])
        for part_info in shape_json['part']:
            mesh_name = Path(part_info['mesh']).stem
            if path_to_latent.get(part_info['mesh']) is None:
                Log.warning(f"Latent code for {mesh_name} not found")
                failed.append(mesh_name)
                continue
            np.savez(output_path / f"{mesh_name}.npz",
                     latent_code=path_to_latent[part_info['mesh']],
                     text=text_to_e_text[f"{shape_name}, {part_info['name']}"])
            success.append(mesh_name)

    # for ex_shape_json_path in list(ex_mesh_info_path.glob('*.json')):
    #     shape_json = json.loads(ex_shape_json_path.read_text())
    #     shape_name = camel_to_snake(shape_json['meta']['model_cat'])
    #     for part_info in shape_json['part']:
    #         mesh_name = Path(part_info['mesh']).stem
    #         if mesh_name in success: continue
    #         if path_to_latent.get(part_info['mesh']) is None:
    #             Log.warning(f"Latent code for {mesh_name} not found")
    #             failed.append(mesh_name)
    #             continue
    #         np.savez(output_path / f"{mesh_name}.npz",
    #                  latent_code=path_to_latent[part_info['mesh']],
    #                  text=text_to_e_text[f"{shape_name}, {part_info['name']}"])
    #         success.append(mesh_name)

    with open(output_path / 'meta.json', 'w') as f:
        json.dump({'ckpt': best_ckpt_path, 'failed': failed, 'success': success}, f, indent=2)

    Log.info('Failed to find latent code for %s', failed)
    Log.info('failed count = %s', len(failed))
    Log.info('success count = %s', len(success))
    Log.info('Done')