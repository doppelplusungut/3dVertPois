import torch
import torch.nn as nn
from monai.networks.blocks.transformerblock import TransformerBlock


class PoiTransformer(nn.Module):
    def __init__(
        self,
        poi_feature_l: int,
        coord_embedding_l: int,
        poi_embedding_l: int,
        vert_embedding_l: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        n_landmarks: int,
        n_verts: int = 22,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = (
            poi_feature_l + coord_embedding_l + poi_embedding_l + vert_embedding_l
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=save_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.n_landmarks = n_landmarks
        self.coord_embedding_l = coord_embedding_l
        self.poi_embedding_l = poi_embedding_l
        self.vert_embedding_l = vert_embedding_l

        self.coordinate_embedding = nn.Linear(3, coord_embedding_l, bias=False)

        self.poi_embedding = nn.Embedding(n_landmarks, poi_embedding_l)
        self.vert_embedding = nn.Embedding(n_verts, vert_embedding_l)

        self.norm = nn.LayerNorm(hidden_size)

        self.fine_pred = nn.Linear(hidden_size, 3)

    def forward(self, coarse_preds, poi_indices, vertebra, poi_features):
        """
        coarse_preds: (B, N_landmarks, 3)
        poi_indices: (B, N_landmarks)
        vertebra: (B)
        poi_features: (B, N_landmarks, poi_feature_l)
        """
        # Create the embeddings
        coords_embedded = self.coordinate_embedding(
            coarse_preds.float()
        )  # size (B, N_landmarks, coord_embedding_l)
        pois_embedded = self.poi_embedding(
            poi_indices
        )  # size (B, N_landmarks, poi_embedding_l)
        vert_embedded = self.vert_embedding(vertebra)  # size (B, 1, vert_embedding_l)

        # Bring vert_embedded to the same shape as the other embeddings
        vert_embedded = vert_embedded.expand(-1, self.n_landmarks, -1)

        # Concatenate the embeddings
        x = torch.cat(
            [poi_features, coords_embedded, pois_embedded, vert_embedded], dim=-1
        )  # size (B, N_landmarks, hidden_size)

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)  # size (B, N, hidden_size)
        x = self.norm(x)
        x = self.fine_pred(x)  # size (B, N, 3)

        return x
