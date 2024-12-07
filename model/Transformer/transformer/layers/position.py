import torch
from .layernorm_gru import LayerNormGRUCell
from torch import nn

class PositionGRUEmbedding(nn.Module):
    def __init__(self, d_model, dim_single_emb, dropout):
        super().__init__()
        self.dim_single_emb = dim_single_emb

        self.fc0_comp = nn.Linear(d_model, d_model)
        self.dropout_comp = nn.Dropout(dropout)
        self.act_comp = nn.ReLU()
        self.fc1_comp = nn.Linear(d_model, dim_single_emb)

        self.combine_fc = nn.Linear(2 * d_model, d_model)

        self.gru = LayerNormGRUCell(d_model, d_model)

    def mlp_compress(self, x):
        x = self.fc0_comp(x)
        x = self.act_comp(x)
        x = self.dropout_comp(x)
        x = self.fc1_comp(x)
        return x

    def forward(self, tokens):
        # ('token'/'fa') * batch * part_idx * attribute_dim
        # tokens
        batch, seq_len, d_model = tokens['token'].size()
        gru_emb = torch.zeros((batch, seq_len, d_model), device=tokens['token'].device)
        fa = tokens['fa']
        for part_idx in range(seq_len):
            # batch, d_model
            hx = gru_emb[torch.arange(batch), fa[:, part_idx], :]
            x = tokens['token'][:, part_idx, :]
            gru_emb[:, part_idx, :] = self.gru(x, hx)

        shorted_gru_emb = self.mlp_compress(gru_emb)

        emb = torch.zeros((batch, seq_len, d_model), device=tokens['token'].device)
        for part_idx in range(seq_len):
            prev = emb[torch.arange(batch), fa[:, part_idx], :-self.dim_single_emb]
            current = shorted_gru_emb[:, part_idx, :]
            emb[:, part_idx, :] = torch.cat((current, prev), dim=-1)

        # print('3 emb', emb.shape)
        # torch.set_printoptions(threshold=2000000)
        # with open('logs/debug/emb0.txt', 'w') as f:
        #     f.write(emb[0].__str__())

        # with open('logs/debug/emb1.txt', 'w') as f:
        #     f.write(emb[1].__str__())

        # with open('logs/debug/emb2.txt', 'w') as f:
        #     f.write(emb[2].__str__())


        tokens = self.combine_fc(torch.cat((tokens['token'], emb), dim=-1))

        return tokens
