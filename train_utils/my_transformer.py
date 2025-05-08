import torch
import torch.nn as nn
# import math

class UserTranformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, dropout: float = 0.1, nhead: int = 2,
                 dim_feedforward: int = 256, num_encoder_layers: int = 2, max_length: int = 100):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()
        self.fc = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # positional encoding
        self.d_model = d_model
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        # transformer encoder
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # tranformer layer
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_encoder_layers)

    def forward(self, x, mask):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        x = self.fc(x).permute(1, 0, 2)

        # add positional encoding to the embeddings
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pe[:x.size(0), :].requires_grad_(False)
        x = self.dropout(x)

        # apply transformer layer
        x = self.transformer_encoder(x, src_key_padding_mask=mask.bool()).permute(1, 0, 2)

        return x



if __name__ == '__main__':

    # 假设每个用户的特征是9维
    input_dim = 9
    # seq_length = 4
    batch_size = 5

    seq_lengths = [8, 6, 10, 7, 9]

    # 构建不同长度的用户特征序列，并进行填充
    max_seq_length = max(seq_lengths)
    padded_sequences = torch.zeros(batch_size, max_seq_length, input_dim)
    for i, seq_len in enumerate(seq_lengths):
        padded_sequences[i, :seq_len, :] = torch.rand(seq_len, input_dim)

    # 创建mask
    src_key_padding_mask = torch.zeros(batch_size, max_seq_length)
    for i, seq_len in enumerate(seq_lengths):
        if seq_len < max_seq_length:
            src_key_padding_mask[i, seq_len:] = 1

    # 定义Transformer的参数
    d_model = 64
    nhead = 2
    num_encoder_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    # 构建一个随机的用户特征序列
    # user_sequence = torch.rand(batch_size, seq_length, input_dim)

    transformer_encoder = UserTranformer(input_dim=input_dim, d_model= d_model, nhead= nhead,
                 dim_feedforward = dim_feedforward, num_encoder_layers= num_encoder_layers, dropout= dropout)

    output = transformer_encoder(padded_sequences, src_key_padding_mask)

    print(output.shape)  # 应该是 (batch_size, seq_length, feature_dim)