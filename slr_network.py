import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import utils
from modules.temporal_layers import BiLSTMLayer, TemporalConv
from modules.visual_extractor import CoSign2s

class KLdis(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T
    
    def forward(self, view1_logits, view2_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        view1_logits = F.log_softmax(view1_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, view2_logits.shape[2] - start_idx)
        ref_probs = F.softmax(view2_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, view2_logits.shape[2] - start_idx)
        loss = self.kdloss(view1_logits, ref_probs)*self.T*self.T
        return loss

class NormBothLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormBothLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(F.normalize(x, dim=-1), F.normalize(self.weight, dim=0))
        return outputs

class TwoStream_Cosign(nn.Module):
    def __init__(self, visual_args, gloss_dict, conv_type, loss_weights, norm_scale=32) -> None:
        super().__init__()
        self.apply_CR = True if 'CR_args' in visual_args else False
        self.visual_module = CoSign2s(**visual_args)
        hidden_size = self.visual_module.out_size
        self.num_classes = len(gloss_dict['id2gloss']) + 1
        self.decoder = utils.Decode(gloss_dict, self.num_classes, 'beam')

        self.stream_configs = {
            'static': {'input_dim': 256*4},
            'motion': {'input_dim': 256*4}, 
            'fusion': {'input_dim': hidden_size}
        }
        for name, config in self.stream_configs.items():
            conv1d = TemporalConv(config['input_dim'], hidden_size, conv_type)
            contextual_module = BiLSTMLayer(
                rnn_type='LSTM', input_size=hidden_size, 
                hidden_size=hidden_size,
                num_layers=2, bidirectional=True
            )
            classifier = NormBothLinear(hidden_size, self.num_classes)
            setattr(self, f'conv1d_{name}', conv1d)
            setattr(self, f'contextual_module_{name}', contextual_module)
            setattr(self, f'classifier_{name}', classifier)

        self.loss = {
          'ctc': torch.nn.CTCLoss(reduction='none', zero_infinity=True),            'kl': KLdis()
        }
        self.loss_weights = loss_weights
        self.norm_scale = norm_scale

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
          if g is not None:  # Safety check
            g[torch.isnan(g)] = 0
            g[torch.isinf(g)] = 0 # Nuke Infinity too

    def forward_contextual(self, framewise, len_x, conv1d_module, contextual_module, classifier):
        conv1d_ret = conv1d_module(framewise.transpose(1,2), len_x)
        conv1d_feat = conv1d_ret['visual_feat'].transpose(0,1) #B,T,C
        feat_len = conv1d_ret['feat_len']
        contextual_feat = contextual_module(conv1d_feat.transpose(0,1), feat_len)['predictions']
        contextual_feat = contextual_feat.transpose(0,1)
        conv1d_logits = classifier(conv1d_feat.transpose(0,1))
        seq_logits = classifier(contextual_feat.transpose(0,1))
        return conv1d_logits, seq_logits, feat_len

    def forward(self, inputs_dict):
        x, len_x = inputs_dict['x'], inputs_dict['len_x']
        visual_ret = self.visual_module(x, len_x)
        if self.apply_CR and self.training:
            results = {}
            for stream_type in self.stream_configs.keys():
                view1, view2 = visual_ret[f'view1_{stream_type}'], visual_ret[f'view2_{stream_type}']
                conv1d_module = getattr(self, f'conv1d_{stream_type}')
                contextual_module = getattr(self, f'contextual_module_{stream_type}')
                classifier = getattr(self, f'classifier_{stream_type}')
                results[f'view1_{stream_type}'] = self.forward_contextual(view1, len_x, conv1d_module, contextual_module, classifier)
                results[f'view2_{stream_type}'] = self.forward_contextual(view2, len_x, conv1d_module, contextual_module, classifier)
            results['feat_len'] = results['view1_static'][-1]
            return results
        else:
            fusion = visual_ret['fusion']
            conv1d_logits_fusion, seq_logits_fusion, feat_len = self.forward_contextual(fusion, len_x, self.conv1d_fusion, self.contextual_module_fusion, self.classifier_fusion)

            def decode_if_not_training(logits):
                return None if self.training else self.decoder.decode(
                    logits*self.norm_scale, feat_len, batch_first=False, probs=False
                )

            return {
                'conv_sents_fusion': decode_if_not_training(conv1d_logits_fusion),
                'recognized_sents_fusion': decode_if_not_training(seq_logits_fusion),
            }

    def get_ctc_loss(self, no_scale_logits, label, feat_len, label_len):
        return self.loss['ctc'](
                        (no_scale_logits*self.norm_scale).log_softmax(-1),
                        label.cpu().int(),
                        feat_len.cpu().int(),
                        label_len.cpu().int(),
                    ).mean()

    def get_loss(self, ret_dict, inputs_dict):
        loss, loss_dict = 0, {}
        label, label_lgt = inputs_dict['label'], inputs_dict['label_lgt']
        
        for k, weight in self.loss_weights.items():
            temp_loss = 0
            
            parts = k.split('_')
            loss_type = parts[1]  
            stream_type = parts[2] 
                
            if loss_type in ['ConvCTC', 'SeqCTC']:
                idx = 0 if loss_type == 'ConvCTC' else 1
                view1_loss = self.get_ctc_loss(ret_dict[f'view1_{stream_type}'][idx], label, ret_dict['feat_len'], label_lgt)
                view2_loss = self.get_ctc_loss(ret_dict[f'view2_{stream_type}'][idx], label, ret_dict['feat_len'], label_lgt)
                temp_loss = (view1_loss + view2_loss) * 0.5 * weight
            else:
                idx = 0 if loss_type == 'Conv' else 1
                view1_logits = ret_dict[f'view1_{stream_type}'][idx] * self.norm_scale
                view2_logits = ret_dict[f'view2_{stream_type}'][idx] * self.norm_scale
                kl_loss1 = self.loss['kl'](view1_logits, view2_logits)
                kl_loss2 = self.loss['kl'](view2_logits, view1_logits)
                temp_loss = (kl_loss1 + kl_loss2) * 0.5 * weight
            
            loss += temp_loss
            loss_dict[k] = temp_loss
            
        return loss, loss_dict