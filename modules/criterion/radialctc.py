import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CTCHelper(object):
    def __init__(self):
        pass

    @staticmethod
    def torch2np(pytensor):
        if torch.is_tensor(pytensor):
            return pytensor.cpu().detach().numpy()
        elif isinstance(pytensor, np.ndarray):
            return pytensor
        else:
            assert "Wrong data type!"

    @staticmethod
    def log_sum_exp(inputs):
        a = max(inputs)
        prob_sum = 0
        for item in inputs:
            prob_sum += np.exp(item - a)
        return np.log(prob_sum) + a

    @staticmethod
    def decode_path(start, paths, label_list):
        decoded_path = list()
        decoded_path.append(start)
        for t in range(paths.shape[0] - 1, 0, -1):
            decoded_path.append(start - paths[t, start])
            start = decoded_path[-1]
        decoded_path = [label_list[idx] for idx in decoded_path[::-1]]
        return decoded_path

    @staticmethod
    def decode_batch_path(start, paths, label_list):
        batch, tlgt, clgt = paths.shape
        decoded_path = np.zeros((paths.shape[:2]), dtype=int)
        decoded_path[:, -1] = start
        for t in range(tlgt - 1, 0, -1):
            try:
                decoded_path[:, t - 1] = start - paths[np.arange(batch), t, start]
            except IndexError:
                pdb.set_trace()
            finally:
                pass
            start = decoded_path[:, t - 1]
        decoded_path = [label_list[i][decoded_path[i]].tolist() for i in range(batch)]
        return decoded_path

    def ctc_forward(self, logits, label, blank=0, operation="sum"):
        append_label = [label[i // 2] if i % 2 == 1 else blank for i in range(len(label) * 2 + 1)]
        len_lgt = len(logits)
        len_label = len(append_label)
        neginf = -1e8
        dp = np.ones((len_lgt, len_label)) * neginf
        paths = np.zeros((len_lgt, len_label), dtype=int)
        dp[0, 0] = logits[0, append_label[0]]
        dp[0, 1] = logits[0, append_label[1]]
        for t in range(1, len_lgt):
            for s in range(0, len_label):
                la1 = dp[t - 1, s]
                if s > 0:
                    la2 = dp[t - 1, s - 1]
                else:
                    la2 = neginf
                if s > 1 and append_label[s] != append_label[s - 2]:
                    la3 = dp[t - 1, s - 2]
                else:
                    la3 = neginf
                if operation == "sum":
                    dp[t, s] = self.log_sum_exp([la1, la2, la3]) + logits[t, append_label[s]]
                else:
                    dp[t, s] = max([la1, la2, la3]) + logits[t, append_label[s]]
                paths[t, s] = np.argmax([la1, la2, la3])
        if operation == "sum":
            return dp, append_label, len_lgt, len_label
        else:
            return dp, append_label, len_lgt, len_label, paths

    def batch_ctc_forward(self, logits, logits_lgt, label, label_lgt, blank=0, operation="sum"):
        batch_size = logits.shape[1]

        append_label_lgt = label_lgt * 2 + 1
        batch_append_label = np.zeros((label.shape[0], append_label_lgt.max().item()), dtype=int)
        batch_append_label[:, 1::2] = label

        temporal_lgt = len(logits)
        max_append_label_lgt = append_label_lgt.max().item()
        neginf = -1e8
        dp = np.ones((batch_size, temporal_lgt, max_append_label_lgt)) * neginf
        paths = np.zeros((batch_size, temporal_lgt, max_append_label_lgt), dtype=int)
        dp[:, 0, 0] = logits[0, np.arange(batch_size), batch_append_label[:, 0]]
        dp[:, 0, 1] = logits[0, np.arange(batch_size), batch_append_label[:, 1]]
        for t in range(1, temporal_lgt):
            for s in range(0, max_append_label_lgt):
                la1 = dp[:, t - 1, s]
                if s > 0:
                    la2 = dp[:, t - 1, s - 1]
                else:
                    la2 = np.ones((batch_size,)) * neginf
                if s > 1:
                    la3 = dp[:, t - 1, s - 2] * (batch_append_label[:, s] != batch_append_label[:, s - 2]) + \
                          np.ones((batch_size,)) * neginf * (batch_append_label[:, s] == batch_append_label[:, s - 2])
                else:
                    la3 = np.ones((batch_size,)) * neginf
                if operation == "sum":
                    dp[t, s] = self.log_sum_exp([la1, la2, la3]) + logits[t, append_label[s]]
                else:
                    dp[:, t, s] = np.maximum(np.maximum(la1, la2), la3) + logits[
                        t, np.arange(batch_size), batch_append_label[:, s]]
                paths[:, t, s] = np.argmax(np.vstack([la1, la2, la3]), axis=0)
        if operation == "sum":
            return dp, append_label, len_lgt, len_label
        else:
            return dp, batch_append_label, temporal_lgt, append_label_lgt, paths

    @staticmethod
    def generate_splits(decoded_path):
        st_idx, ed_idx = 0, 0
        splits = list()
        for idx, lab in enumerate(decoded_path):
            if lab == decoded_path[st_idx]:
                ed_idx += 1
            else:
                splits.append([decoded_path[st_idx], st_idx, ed_idx])
                st_idx = idx
                ed_idx = idx + 1
        if st_idx < len(decoded_path):
            splits.append([decoded_path[st_idx], st_idx, ed_idx])
        assert sum([item[2] - item[1] for item in splits]) == len(decoded_path), f"{max_path}, {splits}"
        return splits

    def decode_max_path(self, log_probs, logits_lgt, labels, label_lgt):
        batch_size = log_probs.shape[1]
        path_list = []
        splits_list = []
        for sample_idx in range(batch_size):
            sample_probs = self.torch2np(log_probs[:logits_lgt[sample_idx], sample_idx])
            sample_labels = labels[sample_idx].tolist()[:label_lgt[sample_idx]]
            dp_max_mat, label_list, lgt, lgt_label, paths = \
                self.ctc_forward(sample_probs, sample_labels, blank=0, operation="max")
            start = lgt_label - 1 - np.argmax(
                [dp_max_mat[lgt - 1, lgt_label - 1], dp_max_mat[lgt - 1, lgt_label - 2]])
            max_path = self.decode_path(start, paths, label_list)
            splits = self.generate_splits(max_path)
            path_list.append(max_path)
            splits_list.append(splits)
        return path_list, splits_list

    def decode_batch_max_path(self, log_probs, logits_lgt, labels, label_lgt):
        if isinstance(logits_lgt, list):
            logits_lgt = np.array(logits_lgt, dtype=int)
        if isinstance(label_lgt, list):
            label_lgt = np.array(label_lgt, dtype=int)

        batch_size = log_probs.shape[1]
        splits_list = []
        dp_max_mat, label_list, lgt, lgt_label, paths = \
            self.batch_ctc_forward(self.torch2np(log_probs), logits_lgt, self.torch2np(labels), label_lgt, blank=0,
                                   operation="max")
        start_idx = lgt_label - 1 - np.argmax(
            np.vstack([
                dp_max_mat[np.arange(batch_size), logits_lgt - 1][np.arange(batch_size), lgt_label - 1],
                dp_max_mat[np.arange(batch_size), logits_lgt - 1][np.arange(batch_size), lgt_label - 2],
            ]), axis=0
        )
        path_list = self.decode_batch_path(start_idx, paths, label_list)
        for sample_idx in range(batch_size):
            splits = self.generate_splits(path_list[sample_idx])
            splits_list.append(splits)
        return path_list, splits_list

    @staticmethod
    def keyframe_cal(logits, info, lambda_func, batch_multiplier=1):
        ind_list = []
        label_list = []
        split_list = info[1] if isinstance(info, tuple) else info
        for idx, splits in enumerate(split_list):
            ind_list += [(logits[item[1]:item[2], idx, item[0]].argmax() + item[1]) * batch_multiplier + idx for item in
                         [*filter(lambda_func, splits)]]
            label_list += [item[0] for item in [*filter(lambda_func, splits)]]
        return ind_list, label_list


