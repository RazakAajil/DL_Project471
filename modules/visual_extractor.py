import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import torchvision.models as models

from .stgcn_layers import Graph, STGCN_block

def generate_mask(shape, part_num, clip_length, ratio, dim):
    """
    Complementary Masks Implementation
    
    Args:
        shape: Input feature shape (B, T, C)
        part_num: Number of groups
        clip_length: Temporal segment length  
        ratio: Masking ratio per clip
        dim: Feature dim per group
    
    Returns:
        two complementary masks (mask_cat_q, mask_cat_k)
    """
    B, T, C = shape
    clips = T // clip_length
    random_mask = np.random.rand(B, clips, part_num) > (1 - 2 * ratio)
    mask_q, mask_k = np.zeros_like(random_mask), np.zeros_like(random_mask)
    position = np.where(random_mask)
    half_num = int(len(position[0]) / 2)

    index = np.random.choice(len(position[0]), half_num, replace=False).tolist()
    for i in range(len(position[0])):
        if i in index:
            mask_q[position[0][i], position[1][i], position[2][i]] = 1
        else:
            mask_k[position[0][i], position[1][i], position[2][i]] = 1
    mask_q = mask_q.astype(np.bool)
    mask_k = mask_k.astype(np.bool)

    mask_cat_q = torch.ones(shape)
    mask_cat_k = torch.ones(shape)
    for i in range(B):
        for k in range(clips):
            if k == clips - 1:
                for j in range(part_num):
                    if mask_q[i, k, j]:
                        mask_cat_q[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        mask_cat_k[i, clip_length*k:, dim * j : dim * (j + 1)] = 0
            else:
                for j in range(part_num):
                    if mask_q[i, k, j]:
                        mask_cat_q[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
                    if mask_k[i, k, j]:
                        mask_cat_k[i, clip_length*k:clip_length*(k+1), dim * j : dim * (j + 1)] = 0
    return mask_cat_q, mask_cat_k

class CoSign1s_block(nn.Module):
    def __init__(self, modes, indims, outdims, A, split, temporal_kernel, adaptive):
        super(CoSign1s_block, self).__init__()
        self.modes = modes
        self.indims = indims
        self.outdims = outdims
        self.A = A
        self.split = split
        self.temporal_kernel = temporal_kernel
        self.gcn_modules = {}
        self.spatial_kernel_size = A[0].size(0)
        self.adaptive = adaptive
        for index, mode in enumerate(self.modes):
            # Group-specific GCN
            self.gcn_modules[mode] = STGCN_block(indims, outdims, (self.temporal_kernel, self.spatial_kernel_size), A[index].clone(), self.adaptive)
        self.gcn_modules = nn.ModuleDict(self.gcn_modules)

    def forward(self, feature):
        index = 0
        feat_list = []
        for mode in self.modes:
            if index == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[index-1], self.split[index]
            if mode == 'hand21':
                hand = self.gcn_modules[mode](torch.cat([feature[:,:,:,start:end], \
                                                      feature[:,:,:,end:self.split[index+1]]]))
                left, right = torch.chunk(hand, 2, dim=0)
                feat_list.append(left)
                feat_list.append(right)
                index += 2
            else:
                feat_list.append(self.gcn_modules[mode](feature[:,:,:,start:end]))
                index += 1
        return torch.cat(feat_list, dim=-1)

class CoSign2s(nn.Module):
    def __init__(self, in_channels, split, temporal_kernel, hidden_size, modes, level, adaptive=True, CR_args=None) -> None:
        super().__init__()
        self.split = split
        self.graph, A = {}, []
        self.part_num = len(self.split)
        self.in_channels = in_channels
        self.modes = modes
        self.CR_args = CR_args
        self.level = level
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f'custom_{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
        self.static_linear = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True)
        )
        self.motion_linear = nn.Sequential(
            nn.Linear(in_channels*2, 64),
            nn.ReLU(inplace=True)
        )
        self.layer_configs = {
            '0': {
                'static': [(64, 64), (64, 128), (128, 256)],
                'motion': [(64, 64), (64, 128), (128, 256)],
                'fusion': [(128, 128), (256, 256), (512, 512)]
            },
            '1': {
                'static': [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                'motion': [(64, 64), (64, 64), (64, 128), (128, 128), (128, 256)],
                'fusion': [(128, 128), (128, 128), (256, 256), (256, 256), (512, 512)]
            }
        }
        
        self.create_layers(A, temporal_kernel, adaptive)

        self.fusion_fusion = nn.Sequential(nn.Linear(512*self.part_num, hidden_size), nn.ReLU(inplace=True))

        self.pool_func = F.avg_pool2d
        self.out_size = hidden_size
        self.final_dim_static = 256
        self.final_dim_motion = 256
        self.final_dim_fusion = 512

    def create_layers(self, A, temporal_kernel, adaptive):
        config = self.layer_configs[self.level]
        
        for layer_type, layer_dims in config.items():
            layers = nn.ModuleList()
            
            for i, (in_dim, out_dim) in enumerate(layer_dims):
                layer_name = self.get_layer_name(layer_type, i)
                
                layer = CoSign1s_block(self.modes, in_dim, out_dim, A, self.split, temporal_kernel, adaptive)
                layers.append(layer)
                setattr(self, layer_name, layer)
            
            setattr(self, f'{layer_type}_layers', layers)

    def get_layer_name(self, layer_type, index):
        if self.level == '0':
            return f'{layer_type}_layer{index + 1}'
        else:
            if index < 4:
                return f'{layer_type}_layer{index // 2 + 1}_{index % 2 + 1}'
            else:
                return f'{layer_type}_layer3'

    def pooling_stage(self, feature):
        feature_list = []
        for i in range(len(self.split)):
            if i == 0:
                start, end = 0, self.split[0]
            else:
                start, end = self.split[i-1], self.split[i]
            feature_list.append(self.pool_func(feature[:,:,:,start:end], \
                                               (1, end - start)).squeeze(-1))
        return torch.cat(feature_list, dim=1)

    def process_static_motion(self, static, motion):
        if self.level == '0':
            processing_steps = [
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat'},
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'},
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'}
            ]
        else:
            processing_steps = [
                {'static_steps': [1, 1], 'motion_steps': [1, 1], 'fusion_steps': [1, 1], 'fusion_input': 'concat'},
                {'static_steps': [1, 1], 'motion_steps': [1, 1], 'fusion_steps': [1, 1], 'fusion_input': 'concat_sum'},
                {'static_steps': [1], 'motion_steps': [1], 'fusion_steps': [1], 'fusion_input': 'concat_sum'}
            ]
        
        static_idx = 0
        motion_idx = 0
        fusion_idx = 0
        
        for step in processing_steps:
            for _ in step['static_steps']:
                static = self.static_layers[static_idx](static)
                static_idx += 1
            
            for _ in step['motion_steps']:
                motion = self.motion_layers[motion_idx](motion)
                motion_idx += 1
            
            if step['fusion_input'] == 'concat':
                fusion_input = torch.cat([static, motion], dim=1)
            else:
                fusion_input = torch.cat([fusion, static + motion], dim=1)
            
            for _ in step['fusion_steps']:
                fusion = self.fusion_layers[fusion_idx](fusion_input)
                fusion_input = fusion
                fusion_idx += 1
        
        return static, motion, fusion

    def apply_masks(self, cat_feat_static, cat_feat_motion, cat_feat_fusion):
        stream_configs = [
            ('static', cat_feat_static, self.final_dim_static),
            ('motion', cat_feat_motion, self.final_dim_motion),
            ('fusion', cat_feat_fusion, self.final_dim_fusion)
        ]
        
        results = {}
        for stream_type, cat_feat, final_dim in stream_configs:
            mask_view1, mask_view2 = generate_mask(
                cat_feat.shape, self.part_num,
                self.CR_args['clip_length'], self.CR_args['ratio'], final_dim
            )
            view1 = mask_view1.to(cat_feat.device) * cat_feat
            view2 = mask_view2.to(cat_feat.device) * cat_feat
            
            if stream_type == 'fusion':
                view1 = self.fusion_fusion(view1)
                view2 = self.fusion_fusion(view2)
            
            results[f'view1_{stream_type}'] = view1
            results[f'view2_{stream_type}'] = view2
        
        return results

    def forward(self, x, len_x):
        if x.shape[3] == 7:
            static = torch.cat([x[:,:,:,0:2], x[:,:,:,6].unsqueeze(-1)], dim=-1)
        else:
            static = x
        static = static[:,:,:,:self.in_channels]
        motion = x[:,:,:,2:6]
        
        static = self.static_linear(static).permute(0,3,1,2) #N,C,T,V
        motion = self.motion_linear(motion).permute(0,3,1,2) #N,C,T,V

        static, motion, fusion = self.process_static_motion(static, motion)

        cat_feat_static = self.pooling_stage(static).transpose(1,2) #B,T,C
        cat_feat_motion = self.pooling_stage(motion).transpose(1,2)
        cat_feat_fusion = self.pooling_stage(fusion).transpose(1,2)

        if self.CR_args is not None and self.training:
            return self.apply_masks(cat_feat_static, cat_feat_motion, cat_feat_fusion)
        else:
            fusion_feat_fusion = self.fusion_fusion(cat_feat_fusion)
            return {
                'fusion': fusion_feat_fusion,
            }  