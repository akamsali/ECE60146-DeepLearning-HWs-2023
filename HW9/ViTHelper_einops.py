import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MasterEncoder_einops(nn.Module):
    def __init__(
        self, max_seq_length, embedding_size, how_many_basic_encoders, num_atten_heads
    ):
        super().__init__()
        # print("creating model...")
        self.max_seq_length = max_seq_length
        self.basic_encoder_arr = nn.ModuleList(
            [
                BasicEncoder(max_seq_length, embedding_size, num_atten_heads)
                for _ in range(how_many_basic_encoders)
            ]
        )  # (A)

    def forward(self, sentence_tensor):
        out_tensor = sentence_tensor
        for i in range(len(self.basic_encoder_arr)):  # (B)
            print("i", i, "out_tensor", out_tensor.shape)
            out_tensor = self.basic_encoder_arr[i](out_tensor)
        return out_tensor


class BasicEncoder(nn.Module):
    def __init__(self, max_seq_length, embedding_size, num_atten_heads):
        super().__init__()
        # print("creating basic encoder...")
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.qkv_size = self.embedding_size // num_atten_heads
        self.num_atten_heads = num_atten_heads
        self.self_attention_layer = SelfAttention(
            max_seq_length, embedding_size, num_atten_heads
        )  # (A)
        self.norm1 = nn.LayerNorm(self.embedding_size)  # (C)
        self.W1 = nn.Linear(
            self.max_seq_length * self.embedding_size,
            self.max_seq_length * 2 * self.embedding_size,
        )
        self.W2 = nn.Linear(
            self.max_seq_length * 2 * self.embedding_size,
            self.max_seq_length * self.embedding_size,
        )
        self.norm2 = nn.LayerNorm(self.embedding_size)  # (E)

    def forward(self, sentence_tensor):
        input_for_self_atten = sentence_tensor.float()
        normed_input_self_atten = self.norm1(input_for_self_atten)
        output_self_atten = self.self_attention_layer(normed_input_self_atten).to(
            device
        )  # (F)
        input_for_FFN = output_self_atten + input_for_self_atten
        normed_input_FFN = self.norm2(input_for_FFN)  # (I)
        basic_encoder_out = nn.ReLU()(
            self.W1(normed_input_FFN.view(sentence_tensor.shape[0], -1))
        )  # (K)
        basic_encoder_out = self.W2(basic_encoder_out)  # (L)
        basic_encoder_out = basic_encoder_out.view(
            sentence_tensor.shape[0], self.max_seq_length, self.embedding_size
        )
        basic_encoder_out = basic_encoder_out + input_for_FFN

        return basic_encoder_out


####################################  Self Attention Code TransformerPreLN ###########################################


class SelfAttention(nn.Module):
    def __init__(self, max_seq_length, embedding_size, num_atten_heads):
        super().__init__()
        # print("creating self attention layer...")
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.qkv_size = embedding_size // num_atten_heads
        self.num_atten_heads = num_atten_heads

        self.to_qvk = nn.Linear(
            self.max_seq_length * self.embedding_size,
            3 * self.max_seq_length * self.embedding_size,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence_tensor):  # (B)
        QKV = self.to_qvk(
            sentence_tensor.reshape(sentence_tensor.shape[0], -1).float()).to(device)
        Q, K, V = tuple(
            rearrange(QKV, "b (k l d h) -> k b h l d", k=3, l=self.max_seq_length,
                d=self.qkv_size, h=self.num_atten_heads,))
        QK_dot_prod = torch.einsum("b h i d , b h j d -> b h i j", Q, K)  # (N)
        attn = self.softmax(QK_dot_prod)
        Z = torch.einsum("b h i j , b h j d -> b h i d", attn, V) / torch.sqrt(
            torch.tensor([self.qkv_size]).float()).to(device)
        return rearrange(Z, "b h l d -> b l (h d)")


# class AttentionHead(nn.Module):
#     def __init__(self, max_seq_length, qkv_size):
#         super().__init__()
#         self.qkv_size = qkv_size
#         self.max_seq_length = max_seq_length
#         self.W = nn.Linear(
#             max_seq_length * self.qkv_size, 3 * max_seq_length * self.qkv_size
#         )  # (B)
#         self.softmax = nn.Softmax(dim=1)  # (E)

#     def forward(self, sentence_portion):  # (F)
#         QKV = self.W(sentence_portion.reshape(sentence_portion.shape[0], -1).float()).to(device)
#         # split into Q, K, V
#         Q, K, V = tuple(rearrange(QKV, 'b (k l d) -> k b l d', k=3, l=self.max_seq_length, d=self.qkv_size))
#         QK_dot_prod = self.softmax(torch.einsum('b i d , b j d -> b i j', Q, K)) # (N)
#         Z = torch.einsum('b i j , b j d -> b i d', QK_dot_prod, V)
#         Z = Z / torch.sqrt(torch.tensor([self.qkv_size]).float()).to(
#             device
#         )
#         return Z
