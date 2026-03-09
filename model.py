"""
STGT-ETD: Spatio-Temporal Graph Transformer for Electricity Theft Detection
Architecture:
  Input → Temporal Embedding → Transformer Encoder →
  GNN Layer → Attention Fusion → FC → Classification
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# ═══════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """
    Injects position information into embeddings.
    Uses sine/cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                          # (T, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()        # (T, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                        # (1, T, d)
        self.register_buffer('pe', pe)

    def forward(self, x):                                           # x: (B, T, d)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ═══════════════════════════════════════════════
# 2. TEMPORAL TRANSFORMER ENCODER
# ═══════════════════════════════════════════════
class TemporalTransformerEncoder(nn.Module):
    """
    Projects raw 1D consumption values → embeddings,
    then applies multi-head self-attention to capture
    long-range temporal dependencies.
    """
    def __init__(self, input_dim: int = 1, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True                                        # (B, T, d)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):                                           # x: (B, T, 1)
        x = self.input_proj(x)                                      # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                                     # (B, T, d_model)
        # Pool over time dimension → consumer embedding
        out = x.mean(dim=1)                                         # (B, d_model)
        return out


# ═══════════════════════════════════════════════
# 3. GRAPH NEURAL NETWORK LAYER
# ═══════════════════════════════════════════════
class GNNLayer(nn.Module):
    """
    Simple Graph Convolutional Network (GCN) layer.
    Aggregates features from neighboring consumers.

    Formula: H' = σ( D^{-1/2} A D^{-1/2} H W )
    where A = adjacency + self-loops, D = degree matrix

    WHY GNN here:
    - Consumers near each other geographically often have similar patterns
    - A thief might appear normal individually but anomalous compared to neighbors
    - GNN captures these spatial/relational dependencies
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear  = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_features)

    def forward(self, H, edge_index, num_nodes):
        """
        H          : (N, in_features)  — node embeddings
        edge_index : (2, E)            — graph connectivity
        num_nodes  : int
        Returns    : (N, out_features)
        """
        # Build sparse adjacency with self-loops
        row, col = edge_index
        # Self-loops
        self_loops = torch.arange(num_nodes, device=H.device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])

        # Degree-based normalization: D^{-1} (simple mean aggregation)
        deg = torch.zeros(num_nodes, device=H.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv = 1.0 / deg.clamp(min=1)

        # Weighted sum of neighbor features
        H_agg = torch.zeros_like(H)
        H_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, H.size(1)),
                           H[row] * deg_inv[row].unsqueeze(1))

        # Linear transform + activation
        H_out = F.relu(self.linear(H_agg))
        H_out = self.dropout(H_out)
        return self.norm(H_out)                                     # (N, out_features)


# ═══════════════════════════════════════════════
# 4. ATTENTION FUSION MODULE
# ═══════════════════════════════════════════════
class AttentionFusion(nn.Module):
    """
    Learns to weight temporal features vs spatial (GNN) features.
    alpha = sigmoid(W * [h_t; h_s])
    fused = alpha * h_t + (1-alpha) * h_s
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, h_temporal, h_spatial):                       # both: (B, d)
        combined = torch.cat([h_temporal, h_spatial], dim=-1)      # (B, 2d)
        alpha    = torch.sigmoid(self.gate(combined))               # (B, d)
        fused    = alpha * h_temporal + (1 - alpha) * h_spatial     # (B, d)
        return fused


# ═══════════════════════════════════════════════
# 5. FULL STGT-ETD MODEL
# ═══════════════════════════════════════════════
class STGT_ETD(nn.Module):
    """
    Spatio-Temporal Graph Transformer for Electricity Theft Detection

    Pipeline:
      x (B, T, 1)
        → TemporalTransformerEncoder → h_t (B, d_model)
        → GNNLayer (using consumer graph) → h_s (B, d_model)
        → AttentionFusion → h_fused (B, d_model)
        → FC → logits (B, 2)

    Args:
        seq_len    : window size (time steps)
        d_model    : embedding dimension (default 64)
        nhead      : attention heads (default 4)
        num_layers : transformer layers (default 2)
        num_nodes  : number of consumers in graph
        dropout    : dropout rate
    """
    def __init__(self, seq_len: int = 30, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 num_nodes: int = 1000, dropout: float = 0.1):
        super().__init__()

        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=1, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            dim_ff=d_model * 4, dropout=dropout
        )
        self.gnn = GNNLayer(d_model, d_model, dropout=dropout)
        self.fusion = AttentionFusion(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)                             # binary: normal / theft
        )

    def forward(self, x, edge_index=None, node_indices=None, num_nodes=None):
        """
        x           : (B, T, 1)          — batch of time windows
        edge_index  : (2, E)             — full consumer graph (optional)
        node_indices: (B,)               — which consumer each sample belongs to
        num_nodes   : int

        If graph info not provided, fusion falls back to temporal only.
        """
        # 1. Temporal encoding
        h_t = self.temporal_encoder(x)                             # (B, d_model)

        # 2. GNN: spatial encoding
        if edge_index is not None and node_indices is not None:
            # Build per-node embedding by averaging windows per consumer
            H = torch.zeros(num_nodes, h_t.size(1), device=x.device)
            counts = torch.zeros(num_nodes, 1, device=x.device)
            H.scatter_add_(0, node_indices.unsqueeze(1).expand_as(h_t), h_t)
            counts.scatter_add_(0, node_indices.unsqueeze(1),
                                torch.ones(len(node_indices), 1, device=x.device))
            H = H / counts.clamp(min=1)

            H_gnn = self.gnn(H, edge_index, num_nodes)             # (N, d_model)
            h_s = H_gnn[node_indices]                              # (B, d_model)
            h_fused = self.fusion(h_t, h_s)
        else:
            h_fused = h_t                                           # temporal only

        # 3. Classify
        logits = self.classifier(h_fused)                          # (B, 2)
        return logits


# ═══════════════════════════════════════════════
# QUICK MODEL SUMMARY
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    model = STGT_ETD(seq_len=30, d_model=64, nhead=4, num_layers=2, num_nodes=100)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] STGT-ETD parameters: {total_params:,}")
    print(model)

    # Dummy forward pass
    x          = torch.randn(8, 30, 1)         # batch=8, T=30, features=1
    edge_index = torch.randint(0, 100, (2, 200))
    node_idx   = torch.randint(0, 100, (8,))
    out        = model(x, edge_index, node_idx, num_nodes=100)
    print(f"[MODEL] Output shape: {out.shape}")  # should be (8, 2)
