import torch
from torch import nn
from rich import print
from .utils.helpers import ResnetBlockFC
from .utils.vq_embedding import VQEmbedding
from .utils.gssoftmax_layer import VQEmbeddingGSSoft
from einops.layers.torch import Rearrange

class TextConditionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.diff_config = config['diffusion_model_paramerter']
        self.t_config = self.diff_config['text_condition']

        seqXcomdmodel = self.t_config['padding_length'] * self.t_config['compressed_d_model']
        self.compress_text_conditon_a = nn.Sequential(*([
                nn.Linear(self.t_config['d_model'], self.t_config['compressed_d_model']),
                Rearrange('b s d -> b (s d)'),
            ]+[
                ResnetBlockFC(seqXcomdmodel) for _ in range(self.t_config['resnet_deepth'])
            ]+[
                nn.Linear(seqXcomdmodel, self.t_config['vq_width'] * self.t_config['vq_height'] * self.t_config['vq_dim_emb']),
                Rearrange('b (h w c) -> b c w h', c=self.t_config['vq_dim_emb'], w=self.t_config['vq_width'], h=self.t_config['vq_height']),
        ]))
        self.compress_text_conditon_b = VQEmbedding(n_e=self.t_config['vq_n_emb'], e_dim=self.t_config['vq_dim_emb'], beta=self.t_config['vq_beta'])
        hwc_total = self.t_config['vq_width'] * self.t_config['vq_height'] * self.t_config['vq_dim_emb']
        self.compress_text_conditon_c = nn.Sequential(*([
                Rearrange('b c w h -> b (h w c)', c=self.t_config['vq_dim_emb'], w=self.t_config['vq_width'], h=self.t_config['vq_height']),
            ]+[
                ResnetBlockFC(hwc_total) for _ in range(self.t_config['resnet_deepth'])
            ]+[
                nn.Linear(hwc_total, self.diff_config['diffusion_model_config']['text_hat_dim']),
        ]))

    def forward(self, text):
        text                    = self.compress_text_conditon_a(text)
        vq_loss, text, _, _, _  = self.compress_text_conditon_b(text)
        text_hat                = self.compress_text_conditon_c(text)

        return vq_loss, text_hat


class ZConditionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_config = config['diffusion_model_paramerter']

        self.mini_resnet_encoder = nn.Sequential(*[
            ResnetBlockFC(self.d_config['dim_latentcode'])
            for _ in range(self.d_config['z_compress_depth'])
        ])

        self.z_hat_dim = self.d_config['diffusion_model_config'].get('z_hat_dim') or self.d_config.get('z_hat_dim')

        # make bottle neck
        self.compress_fc = nn.Linear(self.d_config['dim_latentcode'], self.z_hat_dim * self.d_config['gsemb_latent_dim'])

        self.gssoft_layer = VQEmbeddingGSSoft(
            latent_dim=self.d_config['gsemb_latent_dim'],
            num_embeddings=self.d_config['gsemb_num_embeddings'],
            embedding_dim=self.z_hat_dim
        )

        self.expand_fc = nn.ModuleList([
            nn.Linear(self.z_hat_dim, self.d_config['dim_latentcode'])
            for _ in range(self.d_config['gsemb_latent_dim'])
        ])


    def forward_with_logits_or_x(self, tau, logits=None, x=None):

        # for x (batch, z_hat_dim*4, H=1, W=1) --> (batch, 4, z_hat_dim, H=1, W=1)
        q_z_hat, KL, perplexity, logits = self.gssoft_layer(tau=tau, logits=logits, x=x)

        assert q_z_hat.size(1) == self.d_config['gsemb_latent_dim'] # 4 == 4

        # (batch, 4, z_hat_dim, H=1, W=1) --> (batch, 4, z_hat_dim)
        q_z_hat = q_z_hat.squeeze(-1).squeeze(-1)

        # (batch, 4, z_hat_dim) --> (4, batch, z_hat_dim)
        q_z_hat = q_z_hat.permute(1, 0, 2)

        q_z = []
        for idx, fc in enumerate(self.expand_fc):
            # (batch, z_hat_dim)
            slide_q_z_hat = q_z_hat[idx]
            slide_q_z = fc(slide_q_z_hat)
            q_z.append(slide_q_z)

        # [(batch, z_hat_dim)] * 4 --> (4, batch, z_dim)
        q_z = torch.stack(q_z, dim=0)

        # (4, batch, z_dim) --> (batch, 4, z_dim)
        q_z = q_z.permute(1, 0, 2)

        # print(logits.shape)

        # `M` denote `gsemb_num_embeddings`
        # (batch, 4, M=128, H=1, W=1) --> (batch, 4, M=128)
        logits = logits.squeeze(3).squeeze(3)

        return q_z, KL, perplexity, logits

    def forward(self, z, tau):
        z = self.mini_resnet_encoder(z)
        z_hat = self.compress_fc(z)

        # (batch, z_hat_dim*4) --> (batch, z_hat_dim*4, H=1, W=1)
        z_hat = z_hat.unsqueeze(-1).unsqueeze(-1)

        return self.forward_with_logits_or_x(x=z_hat, tau=tau)





