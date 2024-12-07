from torch import nn

from .feedforward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        m_config = config['transformer_model_paramerter']
        self.n_head = m_config['n_head']
        self.d_model = m_config['d_model']
        self.decoder_dropout = m_config['decoder_dropout']
        self.self_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head,
                                                    dropout=self.decoder_dropout, batch_first=True)

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_head,
                                                     dropout=self.decoder_dropout, batch_first=True)

        self.ffn = PositionWiseFeedForward(d_model=self.d_model, hidden_dim=m_config['ffn_hidden_dim'],
                                           dropout=m_config['ffn_dropout'])

        self.dropout_1 = nn.Dropout(self.decoder_dropout)
        self.norm_1 = nn.LayerNorm(self.d_model)

        self.dropout_2 = nn.Dropout(self.decoder_dropout)
        self.norm_2 = nn.LayerNorm(self.d_model)

        self.dropout_3 = nn.Dropout(self.decoder_dropout)
        self.norm_3 = nn.LayerNorm(self.d_model)

    def forward(self, x, key_padding_mask, attn_mask, enc_data):
        if True:
            before_x = x
            x, attn_weight = self.self_attention(x, x, x,
                                    key_padding_mask=(key_padding_mask == 0),
                                    attn_mask=(attn_mask == 0))

            x = self.dropout_1(x)
            x = self.norm_1(x + before_x)

        if enc_data is not None:
            before_x = x
            # shape of x: (batch, query_len, d_model)
            x, cross_attn_weight = self.cross_attention(x, enc_data, enc_data)

            x = self.dropout_2(x)
            x = self.norm_2(x + before_x)

        if True:
            before_x = x
            x = self.ffn(x)

            x = self.dropout_3(x)
            x = self.norm_3(x + before_x)

        return x, cross_attn_weight