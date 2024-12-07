import torch
import wandb
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import reduce
from torch.nn import functional as F

from .encoder import Encoder
from .decoder import Decoder
from .intermediate import VAE

import utils.mesh as MeshUtils
from utils.mylogging import Log
from utils.base import TransArticulatedBaseModule

class SDFAutoEncoder(TransArticulatedBaseModule):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs

        self.n_validation = 0

        self.e_config = configs["evaluation"]
        self.e_config['eval_mesh_output_path'] = Path(self.e_config['eval_mesh_output_path'] )
        self.e_config['eval_mesh_output_path'].mkdir(parents=True, exist_ok=True)

        # SDF Encoder Decoder Configs
        sdf_configs = configs["SdfModelSpecs"]
        hidden_dim = sdf_configs["hidden_dim"]
        latent_dim = sdf_configs["latent_dim"]
        skip_connection = sdf_configs["skip_connection"]
        tanh_act = sdf_configs["tanh_act"]
        pn_hidden = sdf_configs["pn_hidden_dim"]

        self.encoder = Encoder(c_dim=latent_dim, hidden_dim=pn_hidden, plane_resolution=64)
        self.decoder = Decoder(latent_size=latent_dim, hidden_dim=hidden_dim, skip_connection=skip_connection, tanh_act=tanh_act)

        # VAE Configs
        modulation_dim = latent_dim * 3
        latent_std = configs["latent_std"]
        hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
        self.vae_model = VAE(in_channels=latent_dim * 3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.configs["sdf_lr"])

    def step(self, batch, batch_idx):
        xyz = batch['xyz']
        gt = batch['gt_sdf']
        pc = batch['point_cloud']

        # STEP 1: pointcloud -> triplane features
        plane_features = self.encoder.get_plane_features(pc)

        # STEP 2: triplane features -> z -> triplane features
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, z = out[0], out[-1]

        # STEP 3: triplane features + query points -> SDF
        point_features = self.encoder.forward_with_plane_features(reconstructed_plane_feature, xyz)
        pred_sdf = self.decoder( torch.cat((xyz, point_features),dim=-1) )

        # STEP 4: Loss for VAE and SDF
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.configs["kld_weight"] )
        except Exception as e:
            print(e)
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        return  {"sdf_loss": sdf_loss, "vae_loss": vae_loss, "loss": loss,
                       "reconstructed_plane_feature": reconstructed_plane_feature, 'z': z}

    def training_step(self, batch, batch_idx):
        self.train()
        return_dict = self.step(batch, batch_idx)
        log_dict = {
            'sdf_loss': return_dict['sdf_loss'],
            'vae_loss': return_dict['vae_loss'],
            'loss': return_dict['loss']
        }
        self.log_dict(log_dict, prog_bar=True, enable_graph=False)

        return return_dict["loss"]

    def validation_step(self, batch, batch_idx):
        self.eval()

        return_dict = self.step(batch, batch_idx)
        log_dict = {
            'val_sdf_loss': return_dict['sdf_loss'],
            'val_vae_loss': return_dict['vae_loss'],
            'val_loss': return_dict['loss']
        }
        self.log_dict(log_dict, prog_bar=False, enable_graph=False)

        if batch_idx == 0:
            self.n_validation += 1

        if batch_idx == 0 and self.n_validation % self.e_config['vis_epoch_freq'] == 0:
            batched_recon_latent = return_dict["reconstructed_plane_feature"]
            evaluation_count = min(self.e_config['count'], batched_recon_latent.shape[0])
            screenshots = [np.random.randn(256, 256, 3) * 255 for _ in range(evaluation_count)]
            if self.e_config['count'] > batched_recon_latent.shape[0]:
                Log.warning('`evaluation.count` is greater than batch size. Setting to batch size')
            for batch in tqdm(range(evaluation_count), desc=f'Generating Mesh for Epoch = {batch_idx}'):
                recon_latent = batched_recon_latent[[batch]] # ([1, D*3, resolution, resolution])
                output_mesh = (self.e_config['eval_mesh_output_path'] / f'mesh_{batch_idx}_{batch}.ply').as_posix()
                try:
                    MeshUtils.create_mesh(self, recon_latent,
                                    output_mesh, N=self.e_config['resolution'],
                                    max_batch=self.e_config['max_batch'],
                                    from_plane_features=True)
                    mesh = trimesh.load(output_mesh)
                    screenshot = MeshUtils.generate_mesh_screenshot(mesh)
                except Exception as e:
                    Log.error(f"Error while generating mesh: {e}")
                    if "Surface level must be within volume data range" in str(e):
                        break
                    continue

                screenshots[batch] = screenshot
            image = np.concatenate(screenshots, axis=1)

            self.logger.log_image(key="Image", images=[wandb.Image(image)])

        return return_dict["loss"]
