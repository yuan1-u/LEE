import numpy as np
import torch
from torch import nn
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=True, attention_dropout=0.1, output_attention=False):  # 原scale=None
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=True, attention_dropout=0.1, output_attention=False):  # 原scale=None
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # 维度[batch，头数，序列长度，自动计算值]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 添加一个维度，相当于复制维度，当前维度为[batch，头数，序列长度，序列长度，自动计算值]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # 随机取样，取值范围0~96，取样维度为[序列长度，25]
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        # 96个Q与25个K做计算，维度为[batch，头数，Q个数，K个数，自动计算值]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # 矩阵重组，维度为[batch，头数，Q个数，K个数]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # 分别取到96个Q中每一个Q跟K关系最大的值
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # 在96个Q中选出前25个
        M_top = M.topk(n_top, sorted=False)[1]

        # 取出Q特征，维度为[batch，头数，Q个数，自动计算值]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    # 计算V值
    def _get_initial_context(self, V, L_Q):
        # 取出batch，头数，序列长度，自动计算值
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # 对25个Q以外其他Q的V值，使用平均值(让其继续平庸下去)
            V_sum = V.mean(dim=-2)
            # 先把96个V全部使用平均值代替
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    # 更新25个V值
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:  # 传入的mask_flag是False，为什么是False？
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 计算softmax值
        attn = torch.softmax(scores, dim=-1)

        # 对25个Q更新V，其他仍然为平均值
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):  # 这里的queries,keys, values,attn_mask是从类EncoderLayer的self.attention传入的
        # 取出batch，序列长度，头数，自动计算值
        B, L_Q, H, D = queries.shape
        # 取出序列长度(相当于96个Q，96个K)
        _, L_K, _, _ = keys.shape

        # 维度转置操作，维度变为(batch，头数，序列长度，自动计算值)
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # 选取K的个数，模型核心，用于加速
        # factor为常数5，可以自行修改，其值越大，计算成本越高
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        print('n_top:', u)

        # Q、K选择标准
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print('index:', index)
        print('index.shape:', index.shape)

        # 削弱维度对结果的影响
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # 初始化V值
        context = self._get_initial_context(values, L_Q)
        # 更新25个Q的V值
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        # 取出batch，序列长度，特征数12(即B=32，L=96，_=12)
        B, L, _ = queries.shape
        # 同样的S=96
        _, S, _ = keys.shape
        # 多头注意力机制，这里为8
        H = self.n_heads

        # 通过全连接层将特征512-->512，映射到Q,K,V
        # 512是在进行Embedding后特征数量
        # 同时维度变为(batch，序列长度，多头注意力机制，自动计算)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 计算注意力，  这里的inner_attention是ProbAttention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )  # 调用ProbAttention的forward
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        # 维度batch，序列长度，自动计算值
        out = out.view(B, L, -1)
        # 连接全连接512-->512
        return self.out_projection(out), attn

