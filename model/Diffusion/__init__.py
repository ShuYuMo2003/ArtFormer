import wandb
import torch
import trimesh
import random
import numpy as np
import lightning as L
from pathlib import Path
import utils.mesh as MeshUtils
from torch import nn
from tqdm import tqdm
from utils.base import TransArticulatedBaseModule
from model.SDFAutoEncoder import SDFAutoEncoder

from .diffusion import DiffusionNet
from .diffusion_wapper import DiffusionModel
from .utils.helpers import ResnetBlockFC

from .mini_encoders import TextConditionEncoder, ZConditionEncoder

from utils.mylogging import Log

import os
import json
import yaml


class Diffusion(TransArticulatedBaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.diff_config = config['diffusion_model_paramerter']

        diffusion_core = DiffusionNet(**self.diff_config['diffusion_model_config'])
        self.model = DiffusionModel(diffusion_core, config)

        self.text_mini_encoder = TextConditionEncoder(config)
        self.z_mini_encoder = ZConditionEncoder(config)

        self.e_config = config['evaluation']
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'] )
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)
        try:
            self.sdf = SDFAutoEncoder.load_from_checkpoint(self.e_config['sdf_model_path'])
        except Exception as e:
            print("DO NOT FOUND CUSTOM CKPT. USE DEFAULT. : ", e)
            import time; time.sleep(2)
            absolute_path = Path(os.path.abspath(__file__))
            sdf_config_path = absolute_path.parent.parent.parent / "configs" / "1_SDF" / "train.yaml"
            sdf_config_content = yaml.safe_load(sdf_config_path.read_text())
            self.sdf = SDFAutoEncoder(sdf_config_content)
        self.sdf.eval()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.model.parameters()) +
                                list(self.text_mini_encoder.parameters()) +
                                list(self.z_mini_encoder.parameters()),
                                lr=self.config['lr'])

    def step(self, batch, batch_idx):
        text, z = batch

        max_tau = 0.2 + self.current_epoch * self.config['tau_ratio_on_epoch']
        min_tau = 0.19
        tau = random.uniform(min_tau, max_tau)

        z_conditions, z_KL, z_perplexity, z_logits = self.z_mini_encoder(z, tau)
        vq_loss, text_hat = self.text_mini_encoder(text)

        diff_loss_1, diff_100_loss_1, diff_1000_loss_1, pred_latent_1, perturbed_pc_1 =   \
            self.model.diffusion_model_from_latent(z, cond={
                'z_hat': z_conditions,
                'text': text_hat, # (batch, 4, z_dim)
            })

        z_KL = self.config['z_KL_ratio'] * z_KL

        loss = vq_loss + diff_loss_1 + z_KL

        data = {
            'z': z,
            'pred_latent_1': pred_latent_1,
            'loss': loss,
            'vq_loss': vq_loss,
            'diff_loss_1': diff_loss_1,
            'diff_100_loss_1': diff_100_loss_1,
            'diff_1000_loss_1': diff_1000_loss_1,
            'z_KL': z_KL,
            'z_perplexity': z_perplexity,
            'z_logits': z_logits,
            'tau': tau,
            'tau_max': max_tau,
            'tau_min': min_tau
        }

        return data

    def training_step(self, batch, batch_idx):
        self.train()
        result = self.step(batch, batch_idx)

        del result['pred_latent_1']
        del result['z']
        del result['z_logits']

        self.log_dict(result)

        return result['loss']

    def validation_step(self, batch, batch_idx):
        if batch_idx != 0: return
        self.eval()

        result = self.step(batch, batch_idx)
        pred_latent_1 = result['pred_latent_1']
        z = result['z']

        images = []
        for z in [pred_latent_1, z]:
            # batched_recon_latent = return_dict["reconstructed_plane_feature"]
            batched_recon_latent = self.sdf.vae_model.decode(z) # reconstruced triplane features
            evaluation_count = min(self.e_config['count'], batched_recon_latent.shape[0], z.shape[0])

            screenshots = [np.random.randint(0, 255, (768, 1024, 3)) for _ in range(evaluation_count + 1)]
            if self.e_config['count'] > batched_recon_latent.shape[0]:
                Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')

            for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {batch_idx}'):
                recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
                output_mesh = (self.e_config['eval_mesh_output_path'] / f'mesh_{self.trainer.current_epoch}_{batch}.ply').as_posix()
                try:
                    MeshUtils.create_mesh(self.sdf, recon_latent,
                                    output_mesh, N=self.e_config['resolution'],
                                    max_batch=self.e_config['max_batch'],
                                    from_plane_features=True)
                    mesh = trimesh.load(output_mesh)
                    screenshot = MeshUtils.generate_mesh_screenshot(mesh)
                except Exception as e:
                    Log.error(f"Error while generating mesh: {e}")
                    if "Surface level must be within volume data range" in str(e):
                        continue
                screenshots[batch] = screenshot
            # import pdb; pdb.set_trace();
            image = np.concatenate(screenshots, axis=1)
            images.append(image)
        images = np.concatenate(images, axis=0)

        try:
            self.logger.log_image(key="Image", images=[wandb.Image(images)])
        except Exception as e:
            Log.error(f"Error while logging images: {e}")


