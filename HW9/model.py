from ViTHelper import MasterEncoder
from ViTHelper_einops import MasterEncoder_einops

import torch
import torch.nn as nn
from einops import rearrange


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        embedding_size,
        num_heads,
        num_encoders,
        max_seq_length,
        einops_usage=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.num_patches = (img_size // patch_size) ** 2
        # print(self.num_patches)
        self.patch_embedding = nn.Conv2d(
            3, embedding_size, kernel_size=patch_size, stride=patch_size
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embedding_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        if not einops_usage:
            self.encoder = MasterEncoder(
                max_seq_length=max_seq_length,
                embedding_size=embedding_size,
                how_many_basic_encoders=num_encoders,
                num_atten_heads=num_heads,
            )
        else:
            self.encoder = MasterEncoder_einops(
                max_seq_length=max_seq_length,
                embedding_size=embedding_size,
                how_many_basic_encoders=num_encoders,
                num_atten_heads=num_heads,
            )

        self.mlp_head = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        # flatten  using einops
        x = rearrange(x, "b c h w -> b (h w) c")
        # print(self.cls_token.shape, self.cls_token.expand(x.size(0), -1, -1).shape, x.shape)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = x + self.positional_embedding

        x = self.encoder(x)
        x = self.mlp_head(x[:, 0, :])
        return x