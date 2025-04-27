from torch import nn
import torch
# from model.decoder import Decoder
from model.embed import DataEmbedding
from model.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from model.attn import FullAttention, ProbAttention, AttentionLayer


class Lstm_Informer(nn.Module):
    def __init__(
        self,
        d_ff=256,
        n_heads=32,
        e_layers=2,
        d_feature=1,
        d_mark=4,
        dropout=0.08,
        factor=15,
        distil=True,
        lstm_input_size=4,
        lstm_hidden_size=300,
        lstm_layers=1,
    ):
        super(Lstm_Informer, self).__init__()

        # Attention
        Attn = ProbAttention

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor=factor, attention_dropout=dropout, output_attention=False),
                                   lstm_hidden_size, n_heads, mix=False),
                    lstm_hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(e_layers)
            ],
            [ConvLayer(
                    lstm_hidden_size
            ) for _ in range(e_layers)
            ] if distil else None,
        )

        self.enc_embedding = DataEmbedding(d_mark, lstm_hidden_size, lstm_hidden_size, dropout)
        self.projection = nn.Linear(lstm_hidden_size, d_feature, bias=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, enc_x, enc_mark, enc_self_mask=None):
        device = enc_x.device

        lstm_hidden_size = 300
        lstm_layers = 1
        c0 = torch.randn(lstm_layers, enc_x.shape[0], lstm_hidden_size).float().to(device)
        h0 = torch.randn(lstm_layers, enc_x.shape[0], lstm_hidden_size).float().to(device)
        lstm_out, _ = self.lstm(enc_x, (h0, c0))

        enc_out = self.enc_embedding(lstm_out, enc_mark)

        enc_out1, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        out = self.projection(enc_out1)


        return out
