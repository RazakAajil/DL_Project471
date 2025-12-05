import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    TCN Block with Dilated Convolutions and Residual Connections.
    Modified to be CENTERED (Non-Causal) to align with the downstream BiLSTM.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        
        # 1x1 Conv for residual connection if dimensions change
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type='K-5-P-2-K-5-P-2'):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        # Handle both string and legacy integer input
        if isinstance(conv_type, int):
            # Legacy compatibility - convert integer to string format
            if conv_type == 2:
                self.kernel_size = ['K-5', 'P-2', 'K-5', 'P-2']
            else:
                raise ValueError(f"Unknown integer conv_type: {conv_type}")
        elif isinstance(conv_type, str):
            # Parse the config string (e.g., "K-5-P-2-K-5-P-2")
            self.kernel_size = conv_type.split('-')
            # Group into pairs (e.g., ['K', '5', 'P', '2'] -> ['K-5', 'P-2'])
            self.kernel_size = [f"{self.kernel_size[i]}-{self.kernel_size[i+1]}" 
                               for i in range(0, len(self.kernel_size), 2)]
        else:
            # Assume it's already a list
            self.kernel_size = conv_type

        modules = []
        dilation_level = 0  # Track dilation power (2^0, 2^1, ...)

        for layer_idx, ks in enumerate(self.kernel_size):
            # Handle input dimension changes between layers
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            
            # Parse each kernel spec
            parts = ks.split('-')
            op_type = parts[0]
            
            if op_type == 'P':
                # Pooling layers remain unchanged to preserve 'update_lgt' logic
                modules.append(nn.MaxPool1d(kernel_size=int(parts[1]), ceil_mode=False))
            
            elif op_type == 'K':
                k_size = int(parts[1])
                
                # Calculate dilation for TCN (exponential growth: 1, 2, 4...)
                dilation = 2 ** dilation_level
                dilation_level += 1
                
                # Calculate padding to preserve temporal length (T)
                # Formula for centered same-padding with dilation:
                # P = (D * (K - 1)) / 2
                padding = (dilation * (k_size - 1)) // 2
                
                # Use the robust TemporalBlock (Centered)
                modules.append(
                    TemporalBlock(
                        n_inputs=input_sz, 
                        n_outputs=self.hidden_size, 
                        kernel_size=k_size, 
                        stride=1, 
                        dilation=dilation, 
                        padding=padding
                    )
                )
                
        self.temporal_conv = nn.Sequential(*modules)
        

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            parts = ks.split('-')
            if parts[0] == 'P':
                # Only pooling affects the sequence length
                feat_len = torch.div(feat_len, int(parts[1])).long()
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
        