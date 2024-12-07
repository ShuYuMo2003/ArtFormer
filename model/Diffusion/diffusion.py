# [ArtFormer]: This code is adapted from Diffusion-SDF `https://github.com/princeton-computational-imaging/Diffusion-SDF`
import torch
from torch import nn

from rich import print
from einops import repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from .utils.model_utils import *
from .utils.helpers import *
from utils.mylogging import Log


class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_in_out=None,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        kv_dim = None
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)

        self.kv_dim = kv_dim

        self.layers.append(nn.ModuleList([
                Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention(dim = dim, kv_dim=kv_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention(dim = dim, kv_dim=kv_dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
        self.layers.append(nn.ModuleList([
                Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention(dim = dim, kv_dim=kv_dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))


        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

    def forward(self, x, time_emb=None, context=None):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        assert context is not None
        for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, attn_bias = attn_bias) + x
            x = cross_attn(x, context=context) + x  # removing attn_bias for now

            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim,
        z_hat_dim=None,
        text_hat_dim=None,
        bbox_ratio_dim=None,
        resnet_deepth=None,
        text_expand_ratio=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        self.z_hat_dim = z_hat_dim
        self.text_hat_dim = text_hat_dim
        self.text_expand_ratio = text_expand_ratio

        self.to_time_embeds = nn.Sequential(
            nn.Sequential(SinusoidalPosEmb(self.dim), MLP(self.dim, self.dim)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            # Rearrange('b (n d) -> b n d', n=1)
        )

        self.text_hat_expand = nn.Sequential(*([
                ResnetBlockFC(self.text_hat_dim, dim * self.text_expand_ratio),
            ]+[ ResnetBlockFC(dim * self.text_expand_ratio) for _ in range(resnet_deepth-1)
            ]+[ Rearrange('b (n d) -> b n d', n=self.text_expand_ratio, d=dim)
        ]))

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim))
        self.causal_transformer = CausalTransformer(dim=dim, dim_in_out=self.dim, kv_dim=dim, **kwargs)

    def forward(
        self,
        data,
        diffusion_timesteps,
    ):
        # import pdb; pdb.set_trace()
        assert type(data) is tuple
        assert type(data[1]) is dict
        # adding noise to cond_feature so doing this in diffusion.py
        data, cond = data

        # classifier-free guidance: 40% unconditional
        P = torch.randint(low=0, high=10, size=(1,))
        if P < 2: # 0 1                -> 20% condition free
            cond_feature = {
                'z_hat': torch.zeros_like(cond['z_hat'], device=data.device),
                'text': torch.zeros_like(cond['text'], device=data.device)
            }
        elif P < 4: # 2 3       -> 30% only text condition
            cond_feature = {
                'z_hat': torch.zeros_like(cond['z_hat'], device=data.device),
                'text': cond['text']
            }
        else: #                 -> 60% full condition
            cond_feature = {
                'z_hat': cond['z_hat'],
                'text': cond['text']
            }

        # [<skip>]: z_hat: context-attention to affect the process
        #        text:  cross-attention to affect the process


        batch, dim, device, dtype = *data.shape, data.device, data.dtype

        # import pdb; pdb.set_trace()

        text_hat_expand_condition = self.text_hat_expand(cond_feature['text'])

        z_condition = cond_feature['z_hat'] # (batch, 4, z_dim)
        time_embed = self.to_time_embeds(diffusion_timesteps)
        learned_queries = repeat(self.learned_query, 'd -> b d', b = batch)

        tokens = torch.stack((time_embed, data, learned_queries), dim=1)
        # print(z_condition.shape, tokens.shape)
        tokens = torch.cat((z_condition, tokens), dim=1)

        tokens = self.causal_transformer(tokens, context=text_hat_expand_condition)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[:, -1, :]

        return pred

