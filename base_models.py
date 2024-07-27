import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score
from tqdm import tqdm

class FeaturesCombine(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeaturesCombine, self).__init__()
        self.amp_branch = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.phase_branch = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.combine_layer = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, amp, phase):
        amp_output = self.amp_branch(amp.view(-1,amp.size(-1)))
        phase_output = self.phase_branch(phase.view(-1,phase.size(-1)))
        combined_output = self.combine_layer(torch.cat((amp_output, phase_output), dim=1))
        combined_output=combined_output.view(amp.size(0),amp.size(1),-1)
        return combined_output


class TransformerAutoencoder(nn.Module):        #用
    def __init__(self, input_size, hidden_size, num_heads):
        super(TransformerAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.embedding = nn.Linear(input_size, hidden_size)
        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 将输入形状调整为（seq_len, batch_size, hidden_size）

        attn_output, _ = self.multihead_attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output
        x = self.layer_norm2(x)

        x = x.permute(1, 0, 2)  # 将输出形状调整为（batch_size, seq_len, hidden_size）
        x = self.fc(x)
        return x


# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 定义多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model必须是n_heads的倍数"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        Q = self.split_heads(self.wq(query), batch_size)
        K = self.split_heads(self.wk(key), batch_size)
        V = self.split_heads(self.wv(value), batch_size)

        QK = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            QK = QK.masked_fill(mask == 0, float('-inf'))

        attention_scores = torch.nn.functional.softmax(QK, dim=-1)
        out = torch.matmul(attention_scores, V)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc(out)

        return out


# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attention_output = self.multihead_attention(x, x, x, mask)
        x = x + attention_output
        x = self.layer_norm1(x)

        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.layer_norm2(x)

        return x


# 定义Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, max_len=2000):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads) for _ in range(num_layers)])
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, x, mask):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 定义Transformer解码器（用于重构）
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, input_dim, max_len=2000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, num_layers, max_len=max_len)
        self.fc = nn.Linear(d_model, input_dim)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.transformer(x, mask)
        x = self.fc(x)
        return x

# class TransformerAutoencoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_heads):
#         super(TransformerAutoencoder, self).__init__()
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#
#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
#         self.fc = nn.Linear(hidden_size, input_size)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)  # 将输入形状调整为（seq_len, batch_size, hidden_size）
#
#         attn_output, _ = self.multihead_attention(x, x, x)
#         x = x + attn_output
#
#         x = x.permute(1, 0, 2)  # 将输出形状调整为（batch_size, seq_len, hidden_size）
#         x = self.fc(x)
#         return x


class SeqReconstruction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SeqReconstruction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Complex_spa_temLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(Complex_spa_temLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.maxpool(x)
        # x = x.transpose(1, 2)
        return x

# class Complex_spa_temLayer(nn.Module):
#     def __init__(self,input_size,output_size):
#         super(Complex_spa_temLayer, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=2)
#
#     def forward(self, x):
#         x = self.relu(x)
#         x = x.transpose(1, 2)
#         x = self.maxpool(x)
#         x = x.transpose(1, 2)
#         x=self.fc(x)
#         x = x.transpose(1, 2)
#         return x
