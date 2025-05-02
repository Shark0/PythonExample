import torch
import torch.nn as nn
import math
from positional_encoding import PositionalEncoding

class TransformerRecommender(nn.Module):
    def __init__(self, num_categories, num_article_tags, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerRecommender, self).__init__()
        self.embed_dim = embed_dim

        # 嵌入層：商品種類與文章種類
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.tag_embedding = nn.Embedding(num_article_tags, embed_dim)

        # 位置編碼
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.user_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.article_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 交互層：計算偏好分數
        self.fc = nn.Linear(embed_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_sequence, article_sequence):
        # 嵌入與位置編碼
        user_embed = self.category_embedding(user_sequence) * math.sqrt(self.embed_dim)
        user_embed = self.pos_encoder(user_embed)
        article_embed = self.tag_embedding(article_sequence) * math.sqrt(self.embed_dim)
        article_embed = self.pos_encoder(article_embed)

        # Transformer 編碼
        user_output = self.user_transformer(user_embed.transpose(0, 1)).mean(dim=0)  # 平均池化
        article_output = self.article_transformer(article_embed.transpose(0, 1)).mean(dim=0)  # 平均池化

        # 交互：合併使用者與文章特徵
        combined = torch.cat([user_output, article_output], dim=-1)
        score = self.fc(combined)
        score = self.sigmoid(score)

        return score