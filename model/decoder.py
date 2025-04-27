import torch.nn.functional as F
from torch import nn

from model.attention import ProbAttention, FullAttention

from model.embed import DataEmbedding
from utils.mask import get_attn_subsequence_mask


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout, mix=True)
        self.cross_attention = FullAttention(d_k, d_v, d_model, n_heads, dropout, mix=False)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x, enc_outputs, self_mask=None, cross_mask=None):
        x = self.self_attention(x, x, x, attn_mask=self_mask)
        x = self.cross_attention(x, enc_outputs, enc_outputs, attn_mask=cross_mask)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(x).transpose(-1, 1))
        return self.norm(residual + y)


class Decoder(nn.Module):
    def __init__(
        self, d_k, d_v, d_model, d_ff, n_heads, n_layer, d_mark, dropout, c  # 已改,去掉d_feature
    ):
        super(Decoder, self).__init__()

        self.embedding = DataEmbedding(d_mark, d_model, dropout)  # 已改,去掉d_feature

        self.decoder = nn.ModuleList()  # 是一个nn.ModuleList()类型的列表，存储了解码器的所有层
        for _ in range(n_layer):
            self.decoder.append(
                DecoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout)
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_in, dec_mark, enc_outputs):
        # print('dec_in.shape:', dec_in.shape)  # torch.Size([128, 72, 1])
        # print('dec_mark.shape:', dec_mark.shape)  # torch.Size([128, 72, 4])
        y = self.embedding(dec_in, dec_mark)
        # print('decoder里embedding后的y:', y.shape)  # torch.Size([128, 72, 512])
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(y)
        # print('decoder里get_attn_subsequence_mask后:', dec_self_attn_subsequence_mask.shape)  # torch.Size([128,72,72])

        for layer in self.decoder:  # 每次循环中，当前解码器层的输出y将被赋值给下一次循环中的输入y，以便将其传递给下一层解码器层进行处理
            y = layer(y, enc_outputs, self_mask=dec_self_attn_subsequence_mask)

        # print('decoder里layer后：', y.shape)  # torch.Size([128, 72, 512])
        y = self.norm(y)
        # print('decoder里最终norm后：', y.shape)  # torch.Size([128, 72, 512])

        return y
