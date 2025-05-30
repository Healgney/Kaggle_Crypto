import math as ma
import torch
import torch.nn.functional as F
from torch import nn
from network.Clones import clones

'''
query : [batch_size, time_len, feature_dim]
'''
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / ma.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, local_context_length, device, dropout = 0.1):
        '''
        :param h: number of head
        :param d_model: Features
        :param dropout:
        :param local_context_length:
        :param device: CPU,GPU
        '''
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h #每个 attention head 中 Q、K、V 的维度大小
        self.h = h
        self.local_context_length = local_context_length #窗口大小
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.feature_weight_linear = nn.Linear(d_model, d_model)
        self.device = device

    def forward(self, query, key, value, mask=None, dropout=0.1):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # [batch_size, time_len, feature_dim]
            mask = mask.repeat(query.size()[0], 1, 1)
            mask = mask.to(self.device)
        q_size0 = query.size(0)  # batch
        q_size1 = query.size(1)  # time_len
        q_size2 = query.size(2)  # features
        key_size0 = key.size(0)
        key_size1 = key.size(1)
        key_size2 = key.size(2)
        ##################################query#################################################

        ########################################### local-attention ######################################################

        local_weight_q = torch.matmul(query[:, self.local_context_length - 1:, :],# 未来时间N - 4，全部时间N
                                      query.transpose(-2, -1)) / ma.sqrt(q_size2)  # [batch_size, time_len-4, time_len]
        # [batch_size, time_len-4, time_len]->[batch_size, 1, 5* (time_len-4)]
        local_weight_q_list = [F.softmax(local_weight_q[:, i: i + 1, i: i + self.local_context_length], dim=-1) for i
                               in range(q_size1)] #进行窗口内的softmax，就不需要mask # time_len * [batch_size, 1, window size]
        local_weight_q_list = torch.cat(local_weight_q_list, 2)# 拼接softmax结果
        # [batch_size, 1, 5* (time_len-4)]->[batch_size, 5* (time_len-4), 1]
        local_weight_q_list = local_weight_q_list.permute(0, 2, 1)
        # [batch_size, (time_len-4), feature]->[batch_size, 5* (time_len-4), feature]

        q_list = [query[:, i: i + self.local_context_length, :] for i in range(q_size1-self.local_context_length+1)]
        q_list = torch.cat(q_list, 1)
        # [batch_size, 5* (time_len-4), 1]*[batch_size, 5* (time_len-4), feature]->[batch_size, 5* (time_len-4), feature]
        query = local_weight_q_list * q_list # 时间窗口*时间窗口
        query = q_list
        # [batch_size, 5* (time_len-4), feature]->[batch_size, (time_len-4), 5, feature] # contiguous() 会 重新分配内存，这样view可以直接根据处理好的query改
        query = query.contiguous().view(q_size0, q_size1-self.local_context_length+1, self.local_context_length, q_size2)
        # [batch_size, (time_len-4), 5, feature]->[batch_size, (time_len-4), feature]
        query = torch.sum(query, -2)
        ######################################################################################
        # [[batch_size, head, (time_len-4), feature/head]
        query = query.contiguous().view(q_size0, q_size1, self.h, self.d_k).transpose(1,#分出来head，q_size2 = h*d_k
                                                                                                2)


        ########################################### local-attention ##########################################################################
        local_weight_k = torch.matmul(key[:, self.local_context_length - 1:, :], key.transpose(-2, -1)) / ma.sqrt(
            key_size2)
        local_weight_k_list = [F.softmax(local_weight_k[:, i:i + 1, i:i + self.local_context_length], dim=-1) for i
                               in range(key_size2)]
        local_weight_k_list = torch.cat(local_weight_k_list, 2)
        local_weight_k_list = local_weight_k_list.permute(0, 2, 1)
        k_list = [key[:, i:i + self.local_context_length, :] for i in range(key_size1-self.local_context_length+1)]
        k_list = torch.cat(k_list, 1)
        key = local_weight_k_list * k_list
        key = key.contiguous().view(key_size1, key_size0, self.local_context_length, key_size2, key_size3)
        key = torch.sum(key, 2)

        key = key.contiguous().view(key_size0, key_size1, self.h, self.d_k).transpose(1, 2)
        ##################################################### value matrix #############################################################################
        value = value.view(key_size0 * key_size1, key_size2)  # [4,128,31,2*12]->[4*128,31,2*12]
        nbatches = q_size0 * q_size1
        value = self.linears[0](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # [11*128,31,2,12]

        ################################################ Multi-head attention ##########################################################################
        x, self.attn = attention(query, key, value, mask=None,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = x.view(q_size0, q_size1, q_size2)  # D[11,128,1,2*12] or E[11,128,31,2*12]


        return self.linears[-1](x)


if __name__ == '__main__':
    batch = 1
    time_len = 30
    feature = 3


    q = torch.arange(0, batch*time_len*feature).view(batch, time_len, feature)
    k = torch.arange(0, batch*time_len*feature).view(batch, time_len, feature)
    v = torch.arange(0, batch*time_len*feature).view(batch, time_len, feature)

    mha = MultiHeadedAttention(h=1, d_model = feature, local_context_length = 3, device = 'mps')

    output = mha(q, k ,v)