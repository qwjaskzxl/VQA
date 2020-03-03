import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

import data, config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = config.question_features # default:1024
        vision_features = config.output_features # default:1024
        glimpses = 2

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=config.embedding_features,
            lstm_features=question_features,  # why?
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=config.attn_mid_features,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=config.Classifier_mid_features,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))  #[b, 1024]
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)  # [b, 2048, 14, 14]

        # 得到一个attn分布，其实就是得到img上不同region的重要性
        a = self.attention(v, q) # [b, 2, 14, 14] <- v[b,2048,14,14]  attn  q[b,1024]
        v = apply_attention(v, a) # [b,2*2048]

        combined = torch.cat([v, q], dim=1)# [b,1024+2*2048]
        answer = self.classifier(combined)# [b, num of ans:3000]
        return answer


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        # weight = data.glove_weight(embedding_tokens, embedding_features)
        with h5py.File(config.golve_pretrain_path, 'r') as f:
            weight = torch.FloatTensor(f['weight'])

        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        # self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        # init.xavier_uniform_(self.embedding.weight) #还这样就白pretrain了。print(self.embedding.weight[1])

        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        # print(q, q.shape)
        embedded = self.embedding(q) # [b, max_q_len] -> [b, max_q_len, embed_size]，这里是[b, 23 ,300]
        # print(embedded.shape)
        tanhed = self.tanh(self.drop(embedded)) # [b, 23 ,300]
        packed = pack_padded_sequence(tanhed,  q_len, batch_first=True) #？[]
        _, (_, c) = self.lstm(packed) #lstm的输出：output, hidden, cell, 这里c:[1,b,1024]和hidden维度一样，而output和packed是一样的东西

        return c.squeeze(0) # [1,b,1024] -> [b,1024]


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, kernel_size=1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        # v:[b,2048,14,14], q:[b,lstm_size:1024]
        v = self.v_conv(self.drop(v)) #[b,2048,14,14], 这是1*1 2d卷积，就是在后两维卷
        q = self.q_lin(self.drop(q)) #[b, lstm] -> [b, 设2048]
        q = tile_2d_over_nd(q, v) # [b,2048,14,14] 这里目的就是让q有v的shape （q与v分别对应feature_vector于feature_map）
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x)) # [b,g,14,14]
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. """
    # input:[b, 2048, 14, 14], attention:[b, 2, 14, 14]
    b, c = input.size()[:2] #c:features dim of img
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(b, 1, c, -1)  # [b, 1, c, s] 这里是[b, 1, 2048, 196]
    attention = attention.view(b, glimpses, -1) #[b, g, 196]
    attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [b, g, 1, s], here [n, 2, 1, 196]
    weighted = attention * input  # [b, g, v, s], 其实就是对应元素乘？[b,2,2048,196]
    weighted_mean = weighted.sum(dim=-1)  # [b, g, v], 把最后一维sum out
    return weighted_mean.view(b, -1) #[b, g*v]


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))