import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

# ==============================================================================
# GRAPH CLASS (Required for compatibility with visual_extractor)
# ==============================================================================

class Graph():
    def __init__(self, layout='openpose', strategy='spatial'):
        self.layout = layout
        self.strategy = strategy
        self.get_edge()
        self.hop_dis = 1
        self.get_adjacency()

    def get_edge(self):
        if self.layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                           (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                           (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

    def get_adjacency(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.A = torch.tensor(A, dtype=torch.float32)

# ==============================================================================
# ADAPTIVE ADJACENCY (Matches User's Clean Implementation)
# ==============================================================================

class AdaptiveAdj(nn.Module):
    """
    Learnable Adjacency Mechanism.
    Combines fixed topology with learned attention weights using softmax.
    """
    def __init__(self, A, alpha=0.5):
        super(AdaptiveAdj, self).__init__()
        self.register_buffer('fixed_adj', A)
        self.alpha = alpha
        self.learned_adj = nn.Parameter(torch.zeros_like(A))
        nn.init.xavier_uniform_(self.learned_adj)

    def forward(self):
        # Normalize learned adjacency
        soft_learned = F.softmax(self.learned_adj, dim=-1)
        # Combine with fixed topology
        return self.alpha * self.fixed_adj + (1 - self.alpha) * soft_learned


# ==============================================================================
# GNN VARIANTS (Adapted from User's Code - Superior Logic)
# ==============================================================================

class GCN_unit(nn.Module):
    """Standard GCN Unit with Spatial Partitioning"""
    def __init__(self, out_channels, kernel_size, A, adaptive=True):
        super(GCN_unit, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.adaptive = adaptive
        
        if self.adaptive:
            self.adj = AdaptiveAdj(A)
        else:
            self.register_buffer('A', A)
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        
        A = self.adj() if self.adaptive else self.A
        
        # Graph convolution: einsum for efficiency
        x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        
        return self.relu(self.bn(x))


class GAT_unit(nn.Module):
    """
    Graph Attention Network.
    Uses time-averaged attention for stability (User's logic).
    """
    def __init__(self, out_channels, kernel_size, A, adaptive=True, num_heads=4):
        super(GAT_unit, self).__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.out_channels = out_channels
        
        # Structural mask from fixed adjacency
        self.register_buffer('mask', (A.sum(0) > 0).float())
        
        # Attention parameters - Fixed dimension for multi-head
        head_dim = out_channels // num_heads
        self.W = nn.Conv2d(out_channels, out_channels, 1)
        self.att = nn.Parameter(torch.zeros(1, num_heads, head_dim, 1, 1))
        nn.init.xavier_uniform_(self.att)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        n, kc, t, v = x.size()
        c = kc // self.kernel_size
        x = x.view(n, self.kernel_size, c, t, v)
        
        # Use mean feature across partitions for attention
        feat = x.mean(dim=1)  # (N, C, T, V)
        feat = self.W(feat)
        
        # Compute attention ONCE for spatial graph
        # Average across time for stable attention (User's improvement)
        feat_pooled = feat.mean(dim=2)  # (N, C, V)
        
        # Reshape for multi-head attention
        feat_pooled = feat_pooled.view(n, self.num_heads, c // self.num_heads, v)
        
        # Multi-head attention computation
        f_i = feat_pooled.unsqueeze(-1)  # (N, H, C/H, V, 1)
        f_j = feat_pooled.unsqueeze(-2)  # (N, H, C/H, 1, V)
        
        # Attention scores: ||W*h_i + W*h_j||
        scores = self.leaky_relu(f_i + f_j)  # (N, H, C/H, V, V)
        scores = (scores * self.att).sum(dim=(1,2), keepdim=False)  # (N, V, V)
        
        # Mask with structural prior
        scores = scores.masked_fill(self.mask.unsqueeze(0) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)  # (N, V, V)
        
        # Apply attention to all timesteps (broadcast across time)
        # (N, K, C, T, V) @ (N, V, V) -> (N, K, C, T, V)
        out = torch.einsum('nkctv,nvw->nkctw', x, attn)
        out = out.sum(dim=1)  # Aggregate partitions
        
        return self.relu(self.bn(out))


class GIN_unit(nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, out_channels, kernel_size, A, adaptive=True):
        super(GIN_unit, self).__init__()
        self.kernel_size = kernel_size
        self.adaptive = adaptive
        
        if self.adaptive:
            self.adj = AdaptiveAdj(A)
        else:
            self.register_buffer('A', A)
            
        self.eps = nn.Parameter(torch.zeros(1))
        
        # MLP for GIN
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, kc, t, v = x.size()
        c = kc // self.kernel_size
        x = x.view(n, self.kernel_size, c, t, v)
        
        A = self.adj() if self.adaptive else self.A
        
        # Aggregate neighbors
        neighbor_agg = torch.einsum('nkctv,kvw->nctw', (x, A))
        self_feat = x.sum(dim=1)
        
        # GIN formula
        out = (1 + self.eps) * self_feat + neighbor_agg
        out = self.mlp(out)
        
        return self.relu(self.bn(out))


class SAGE_unit(nn.Module):
    """GraphSAGE with mean aggregation"""
    def __init__(self, out_channels, kernel_size, A, adaptive=True):
        super(SAGE_unit, self).__init__()
        self.kernel_size = kernel_size
        self.adaptive = adaptive
        
        if self.adaptive:
            self.adj = AdaptiveAdj(A)
        else:
            self.register_buffer('A', A)
            
        self.conv_self = nn.Conv2d(out_channels, out_channels, 1)
        self.conv_neigh = nn.Conv2d(out_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, kc, t, v = x.size()
        c = kc // self.kernel_size
        x = x.view(n, self.kernel_size, c, t, v)
        
        A = self.adj() if self.adaptive else self.A
        
        # Mean aggregation
        neigh_feat = torch.einsum('nkctv,kvw->nctw', (x, A))
        self_feat = x.sum(dim=1)
        
        # Combine
        out = self.conv_self(self_feat) + self.conv_neigh(neigh_feat)
        
        return self.relu(self.bn(out))


# ==============================================================================
# GNN ENSEMBLE MODULE
# ==============================================================================

class GNNEnsemble(nn.Module):
    """
    Ensemble of all 4 GNN types with learnable weights.
    Processes input through all GNNs and combines outputs.
    """
    def __init__(self, out_channels, kernel_size, A, adaptive=True, ensemble_method='attention'):
        super(GNNEnsemble, self).__init__()
        
        # Create all 4 GNN types
        self.gcn = GCN_unit(out_channels, kernel_size, A.clone(), adaptive)
        self.gat = GAT_unit(out_channels, kernel_size, A.clone(), adaptive)
        self.gin = GIN_unit(out_channels, kernel_size, A.clone(), adaptive)
        self.sage = SAGE_unit(out_channels, kernel_size, A.clone(), adaptive)
        
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'attention':
            # Learnable attention weights for each GNN
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels * 4, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, 4, 1)  # 4 weights for 4 GNNs
            )
        elif ensemble_method == 'weighted_sum':
            # Simple learnable weights
            self.weights = nn.Parameter(torch.ones(4) / 4)
        elif ensemble_method == 'average':
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
    def forward(self, x):
        # Pass through all GNNs
        gcn_out = self.gcn(x)
        gat_out = self.gat(x)
        gin_out = self.gin(x)
        sage_out = self.sage(x)
        
        if self.ensemble_method == 'attention':
            # Concatenate all outputs
            concat_out = torch.cat([gcn_out, gat_out, gin_out, sage_out], dim=1)
            
            # Compute attention weights
            attn_weights = self.attention(concat_out)  # (N, 4, T, V)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Stack outputs and apply weighted sum
            stacked = torch.stack([gcn_out, gat_out, gin_out, sage_out], dim=1)  # (N, 4, C, T, V)
            attn_weights = attn_weights.unsqueeze(2)  # (N, 4, 1, T, V)
            
            out = (stacked * attn_weights).sum(dim=1)
            
        elif self.ensemble_method == 'weighted_sum':
            # Normalize weights
            weights = F.softmax(self.weights, dim=0)
            
            # Weighted sum
            out = (weights[0] * gcn_out + 
                   weights[1] * gat_out + 
                   weights[2] * gin_out + 
                   weights[3] * sage_out)
            
        else:  # average
            out = (gcn_out + gat_out + gin_out + sage_out) / 4
            
        return out


# ==============================================================================
# STGCN BLOCK WITH ENSEMBLE (Uses GNN Ensemble + Original Temporal Logic)
# ==============================================================================

class STGCN_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        A,
        adaptive=True,
        stride=1,
        dropout=0,
        residual=True,
        gnn_type='ensemble',  # Now defaults to ensemble
        ensemble_method='attention'  # Method for combining GNNs
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        
        # 1. 1x1 Conv for Channel Expansion (Input to GNN)
        self.conv_1x1 = nn.Conv2d(
            in_channels,
            out_channels * kernel_size[1],
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )

        # 2. Select GNN Backbone or Ensemble
        gnn_type = gnn_type.lower()
        if gnn_type == 'ensemble':
            # Use ensemble of all GNNs
            self.gcn = GNNEnsemble(out_channels, kernel_size[1], A, adaptive, ensemble_method)
        elif gnn_type == 'gcn':
            self.gcn = GCN_unit(out_channels, kernel_size[1], A, adaptive=adaptive)
        elif gnn_type == 'gat':
            self.gcn = GAT_unit(out_channels, kernel_size[1], A, adaptive=adaptive)
        elif gnn_type == 'gin':
            self.gcn = GIN_unit(out_channels, kernel_size[1], A, adaptive=adaptive)
        elif gnn_type == 'sage':
            self.gcn = SAGE_unit(out_channels, kernel_size[1], A, adaptive=adaptive)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # 3. Temporal Modeling (Reverted to Original ST-GCN logic)
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # 4. Residual Connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        res = self.residual(x)
        
        # 1. Expand channels for K partitions
        x = self.conv_1x1(x)
        
        # 2. Apply GNN or GNN Ensemble (spatial)
        x = self.gcn(x)
        
        # 3. Apply TCN (temporal)
        x = self.tcn(x)
        
        return self.relu(x + res)


# ==============================================================================
# CHAINS (Standard Interface for visual_extractor.py)
# ==============================================================================

class STGCNChain(nn.Sequential):
    def __init__(self, in_dim, block_args, kernel_size, A, adaptive, gnn_type='ensemble', ensemble_method='attention'):
        super(STGCNChain, self).__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(f'layer{i}_{j}', STGCN_block(
                    last_dim, channel, kernel_size, A.clone(), adaptive, 
                    gnn_type=gnn_type, ensemble_method=ensemble_method
                ))
                last_dim = channel

def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive, gnn_type='ensemble', ensemble_method='attention'):
    """
    Create STGCN chain with specified configuration.
    
    Args:
        in_dim: Input dimension
        level: Complexity level ('0', '1', or '2')
        kernel_size: Kernel size for convolutions
        A: Adjacency matrix
        adaptive: Whether to use adaptive adjacency
        gnn_type: 'ensemble', 'gcn', 'gat', 'gin', or 'sage'
        ensemble_method: 'attention', 'weighted_sum', or 'average' (only used when gnn_type='ensemble')
    """
    if level == '0':
        block_args = [[64,1], [128,1], [256,1]]
    elif level == '1':
        block_args = [[64,2], [128,2], [256,1]]
    elif level == '2':
        block_args = [[128,1], [256,1], [512,1]]
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive, gnn_type, ensemble_method), block_args[-1][0]