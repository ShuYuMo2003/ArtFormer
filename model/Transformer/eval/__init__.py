# [ArtFormer]: This file contains the class to make the inference (generation of articulated object) from text or image condition.
import os
import copy
import torch
import json
import time
import random
import pickle

from pathlib import Path

import torch.utils
import torch.nn.functional as F
from tqdm import trange
# from rich import print
from transformers import AutoTokenizer, T5EncoderModel
from ..dataloader import TransDiffusionDataset
from .. import TransDiffusionCombineModel
from model.SDFAutoEncoder import SDFAutoEncoder

from utils import untokenize_part_info, generate_gif_toy, fit_into_bounding_box
from utils.por_cuda import POR
import utils.mesh as MeshUtils
from utils.mylogging import Log
from utils.z_to_mesh import GenSDFLatentCodeEvaluator

import sys
sys.path.append('../../..')
from eval.visualize import visualize_obj_high_q

class Evaluater():
    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.device = eval_config['device']
        self.number_of_trial = self.eval_config['number_of_trial']

        Log.info("Loading model %s", TransDiffusionCombineModel)
        self.model = TransDiffusionCombineModel.load_from_checkpoint(eval_config['checkpoint_path'])
        self.model.eval()
        # self.model.diffusion.model.cond_dropout = False
        self.m_config = self.model.config

        self.z_mini_encoder = self.model.z_mini_encoder

        d_configs = self.m_config['dataset_n_dataloader']

        self.dataset = TransDiffusionDataset(dataset_path=d_configs['dataset_path'],
                cut_off=d_configs['cut_off'],
                enc_data_fieldname=d_configs['enc_data_fieldname'],
                cache_data=False)

        self.eval_output_path = Path(self.eval_config['eval_output_path']) / time.strftime("%m-%d-%I%p-%M-%S")
        os.makedirs(self.eval_output_path, exist_ok=True)

        self.start_token = copy.deepcopy(self.dataset.start_token).to(self.device)
        self.end_token = copy.deepcopy(self.dataset.end_token).to(self.device)

        Log.info("Loading model %s", T5EncoderModel)
        self.tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache')
        self.text_encoder = T5EncoderModel.from_pretrained('google-t5/t5-large', cache_dir='cache/t5_cache').to(self.device)
        #TODO: check need to do self.text_encoder.eval() or not
        self.text_encoder.eval()
        self.t5_max_sentence_length = self.eval_config['t5_max_sentence_length']

        # self.equal_part_threshold = self.eval_config['equal_part_threshold']

        # self.latentcode_evaluator = LatentCodeEvaluator(Path(self.dataset.get_onet_ckpt_path()), 100000, 16, self.device)

        Log.info("Loading model %s", SDFAutoEncoder)
        self.gensdf_config = self.eval_config['gensdf_latentcode_evaluator']
        # self.gensdf_config['gensdf_model_path'] = self.dataset.get_best_sdf_ckpt_path()
        self.sdf = self.model.sdf # SDFAutoEncoder.load_from_checkpoint(self.gensdf_config['gensdf_model_path'])
        self.sdf.eval()
        self.latentcode_evaluator = GenSDFLatentCodeEvaluator(self.sdf, eval_mesh_output_path=self.eval_output_path,
                                                             resolution=self.gensdf_config['resolution'],
                                                             max_batch=self.gensdf_config['max_batch'],
                                                             device=self.device)

    def encode_text(self, text):
        input_ids = self.tokenizer([text], return_tensors="pt", padding='max_length',
                                    max_length=self.t5_max_sentence_length).input_ids
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
        encoded_text = outputs.last_hidden_state.detach()
        return encoded_text

    def generate_non_padding_mask(self, len):
        return torch.ones(1, len).to(self.device)

    def is_end_token(self, token):
        length = token.size(0)
        difference = torch.nn.functional.mse_loss(token[:length], self.end_token[:length])
        Log.info('    - Difference with end token: %s', difference.item())
        return difference < self.equal_part_threshold

    def inference_from_text(self, text, enc_data=None, need_mesh=True):
        Log.info('[1] Inference text: %s', len(text))
        if enc_data is None:
            encoded_text = self.encode_text(text)
        else:
            encoded_text = enc_data.unsqueeze(0).to(self.device)

        exist_node = {
            'fa': torch.tensor([0]).to(self.device),
            'token': copy.deepcopy((self.start_token[:16])).unsqueeze(0).to(self.device),
            'text_hat': torch.zeros((64)).unsqueeze(0).to(self.device),
            'z_hat': torch.zeros((4, 768)).unsqueeze(0).to(self.device),
            'latent': torch.zeros((768)).unsqueeze(0).to(self.device)
        }
        round = 1
        Log.info('[2] Generate nodes')
        atten_weights_list = []

        use_shape_prior = True
        while True:
            current_length = exist_node['token'].size(0)
            Log.info('   - Generate nodes round: %s, part count: %s', round, exist_node['token'].size(0))
            with torch.no_grad():
                # input: (batch, seq, xxx) ---> (batch|seq, xxx) base on `padding_mask`, the dimension of batch & seq are merged.
                # batch=1 for evaluation.
                output = self.model.transformer({
                                'fa': exist_node['fa'].unsqueeze(0),        # batched.
                                'token': torch.cat((exist_node['token'], exist_node['text_hat']), dim=1).unsqueeze(0),
                            },
                            self.generate_non_padding_mask(current_length),
                            encoded_text) # unbatched.
            atten_weights_list.append(output['cross_attn_weight_list'])
            # Solve End Token.
            # True -> not end token, False -> end token
            end_token_mask = output['is_end_token_logits'] > 0
            Log.info('   - Check end token: %s', output['is_end_token_logits'])
            Log.info('   - Check end token mask: %s', end_token_mask)
            if not torch.any(end_token_mask):
                break

            articulated_info = output['articulated_info'][end_token_mask]

            condition = output['condition']
            if isinstance(condition, dict):
                pred_text_hat = condition['text_hat'][end_token_mask] # torch.Size([1, 64])
                pred_z_logits = condition['z_logits'][end_token_mask] # torch.Size([1, 4, 128])
                q_z, _KL, _perplexity, _logits = self.z_mini_encoder.forward_with_logits_or_x(tau=0.5, logits=pred_z_logits)
                latent_code = None
            else:
                latent_code = condition[end_token_mask] # torch.Size([1, 768])
                pred_text_hat = torch.zeros((64)).unsqueeze(0).to(self.device)
                q_z = None
                use_shape_prior = False

            result = articulated_info

            fa_idx = torch.arange(end_token_mask.shape[0], device=self.device)
            fa_idx = fa_idx[end_token_mask]

            exist_node['fa'] = torch.cat((exist_node['fa'], fa_idx), dim=0)
            exist_node['token'] = torch.cat((exist_node['token'], result), dim=0)

            if pred_text_hat is not None:   exist_node['text_hat'] = torch.cat((exist_node['text_hat'], pred_text_hat), dim=0)
            if q_z is not None:             exist_node['z_hat'] = torch.cat((exist_node['z_hat'], q_z), dim=0)
            if latent_code is not None:     exist_node['latent'] = torch.cat((exist_node['latent'], latent_code), dim=0)


        Log.info('[3] reconstruct latent code with condition')
        if use_shape_prior:
            latent = self.model.diffusion.model.generate_conditional({
                'z_hat': exist_node['z_hat'],
                'text': exist_node['text_hat'],
            })
            exist_node['token'] = torch.cat((exist_node['token'], latent), dim=-1)
        else:
            exist_node['token'] = torch.cat((exist_node['token'], exist_node['latent']), dim=-1)

        processed_nodes = []
        Log.info('[4] Generate mesh')

        for idx in trange(exist_node['fa'].shape[0], desc='   - Generate mesh'):
            dfn_fa = exist_node['fa'][idx].item()
            token  = exist_node['token'][idx].cpu().tolist()
            processed_node = {
                'dfn': idx,
                'dfn_fa': dfn_fa,
            }
            part_info = untokenize_part_info(token)

            z = torch.tensor(part_info['latent_code']).to(self.device)
            if need_mesh: part_info['mesh'] = self.latentcode_evaluator.generate_mesh(z.unsqueeze(0))
            # import pdb; pdb.set_trace()
            part_info['z'] = z
            # raw_points_sdf, rho = self.latentcode_evaluator.generate_uniform_point_cloud_inside_mesh(z.unsqueeze(0))
            # part_info['points'], part_info['rho'] = fit_into_bounding_box(raw_points_sdf, rho, part_info['bbx'])

            processed_node.update(part_info)
            processed_nodes.append(processed_node)

        # import pdb; pdb.set_trace()

        # We do not want start token.
        return processed_nodes[1:], atten_weights_list

    def inference_to_output_path(self, text, output_path, enc_data=None, blender_generated_gif=False):
        output_path.mkdir(exist_ok=True, parents=True)
        processed_nodes, atten_weights_list = self.inference_from_text(text, enc_data)

        # for debug only.
        # processed_nodes, atten_weights_list = pickle.load(open('/ssd1/dengzhidong/.sym/final/ArtFormer/elog/Final_OP1_05-27-01PM-29-48/StorageFurniture_45243_1/0/output.dat', 'rb')), None

        output_data_path = output_path / "output.dat"
        with open(output_data_path, 'wb') as f: f.write(pickle.dumps(processed_nodes))
        Log.info("[Write] %s", output_data_path)

        output_tex_path = output_path / "input.txt"
        output_tex_path.write_text(text)
        Log.info("[Write] %s", output_tex_path)

        output_gif_path = output_path / "gif"
        generate_gif_toy(processed_nodes, output_gif_path, bar_prompt="   - Generate Frames", blender_generated_gif=blender_generated_gif)
        Log.info("[Write] %s", output_gif_path)

        # output_temp_path : Path = output_path / "temp"
        # output_temp_path.mkdir(exist_ok=True, parents=True)
        # Log.info("[Write] %s", output_temp_path)

        # for ratio in [0, 0.5, 1]:
        #     visualize_obj_high_q(processed_nodes, output_temp_path / str(ratio), output_path / str(ratio), ratio)

        # return atten_weights_list

    def inference_dat_file_only(self, text, output_dat_path, enc_data=None):
        processed_nodes, atten_weights_list = self.inference_from_text(text, need_mesh=False, enc_data=enc_data)
        with open(output_dat_path, 'wb') as f:
            f.write(pickle.dumps(processed_nodes))

    def inference(self, text):
        number_of_trial = self.number_of_trial
        list_processed_nodes = [None] * number_of_trial
        for trial in trange(number_of_trial, desc="Doing trial"):
            processed_nodes = self.inference_from_text(text)
            list_processed_nodes[trial] = {
                'data': processed_nodes,
                'rate': POR(processed_nodes, n_sample=8192),
            }
            rate = list_processed_nodes[trial]['rate']
            output_gif_path = (Path(self.eval_output_path) / f'output_{trial}_{rate}.gif')
            Log.info('[4] Generate Gif: %s', output_gif_path.as_posix())

            generate_gif_toy(processed_nodes, output_gif_path,
                            bar_prompt="   - Generate Frames")
            Log.info('[5] Done')

        output_json_path = (Path(self.eval_output_path) / f'output.json')
        output_json_path.write_text('{"text": "' + text + '"}')

        output_data_path = (Path(self.eval_output_path) / f'output.data')
        with open(output_data_path, 'wb') as f:
            f.write(pickle.dumps(list_processed_nodes))
        Log.info("Saved data checkpoint %s.", output_data_path.as_posix())