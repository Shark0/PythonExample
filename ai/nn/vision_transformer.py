import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, num_classes=10, dim=64, depth=2, heads=4, mlp_dim=128):
        super(VisionTransformer, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        # Patch 嵌入
        self.patch_embed = nn.Linear(patch_dim, dim)
        # 位置編碼
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 分類標記（CLS token）
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # 分類頭
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 將圖像分割為 patch 並展平
        x = x.view(batch_size, 1, 28, 28)
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # 分割為 7x7 個 4x4 patch
        x = x.contiguous().view(batch_size, 49, 16)  # (batch, num_patches, patch_dim)
        # Patch 嵌入
        x = self.patch_embed(x)  # (batch, num_patches, dim)
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_patches+1, dim)
        # 添加位置編碼
        x = x + self.pos_embed
        # Transformer 編碼
        x = self.transformer(x)
        # 取出 CLS token 的輸出進行分類
        x = x[:, 0]
        x = self.mlp_head(x)
        return x