import torch
import random
import numpy as np

EPS = 1e-4


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, skeleton):
        for t in self.transforms:
            skeleton = t(skeleton)
        return skeleton


class ToTensor(object):
    def __call__(self, skeleton):
        if isinstance(skeleton, np.ndarray):
            skeleton = torch.from_numpy(skeleton).float()
        return skeleton

class Jitter(object):
    """
    Apply Gaussian jitter (noise) to skeleton sequences.
    
    Args:
        std_dev (float): Standard deviation of the Gaussian noise.
    """
    
    def __init__(self, std_dev=0.01) -> None:
        self.std_dev = std_dev

    def __call__(self, skeleton):
        noise = np.random.normal(loc=0, scale=self.std_dev, size=skeleton.shape)
        return skeleton + noise

class TemporalDropout(object):
    """
    Apply temporal dropout by randomly removing a contiguous segment of frames.
    
    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """
    
    def __init__(self, max_dp=0.2) -> None:
        self.max_dp = max_dp     

    def __call__(self, clip):
        vid_len = len(clip)
        dp_len = int(vid_len * self.max_dp * np.random.random())
        start = np.random.randint(0, vid_len - dp_len + 1)
        end = start + dp_len
        index = list(range(0, start)) + list(range(end, vid_len))

        new_len = vid_len - dp_len
        start_idx = 0 if random.uniform(0,1) > 0.5 else 1
        index_ = list(range(start_idx, new_len, 2))
        index_rgb = [index[num] for num in index_]
        return clip[index_rgb]

class TemporalCrop(object):
    """
    Apply temporal cropping by dropping frames from the beginning and end.
    
    Args:
        max_dp (float): Maximum dropout proportion. Actual dropout length
            is between [0, vid_len * max_dp].
    """
    
    def __init__(self, max_dp=0.2) -> None:
        self.max_dp = max_dp

    def __call__(self, clip):
        vid_len = len(clip)
        dp_len = int(vid_len * self.max_dp * np.random.random())
        drop_head = random.randint(0, dp_len)
        drop_tail = dp_len - drop_head

        start_idx = drop_head
        end_idx = vid_len - drop_tail      
        index = list(range(start_idx, end_idx))     

        new_len = vid_len - dp_len
        start_idx = 0 if random.uniform(0,1) > 0.5 else 1
        index_ = list(range(start_idx, new_len, 2))
        index_rgb = [index[num] for num in index_]
        return clip[index_rgb]

class Dropout_kp(object):
    """
    Apply dropout to skeleton keypoints.
    
    Args:
        drop_prob (float): Probability of dropping each keypoint at each frame.
    """
    
    def __init__(self, drop_prob=0.1) -> None:
        self.drop_prob = drop_prob    

    def __call__(self, skeleton):
        T, K, _ = skeleton.shape
        mask = np.random.rand(T, K) > self.drop_prob
        return skeleton * mask[..., np.newaxis]
class Spatial_flip(object):
    """
    Apply spatial flipping to skeleton sequences.
    
    Args:
        prob (float): Probability of applying spatial flip.
    """
    
    def __init__(self, prob=0.5) -> None:
        self.prob = prob

    def __call__(self, skeleton):
        flag = random.random() < self.prob
        if flag:
            flipped_skeleton = skeleton.copy()
            flipped_skeleton[..., 0] = - flipped_skeleton[..., 0]
            flipped_skeleton[:, 0:21], flipped_skeleton[:, 21:42] = flipped_skeleton[:, 21:42].copy(), flipped_skeleton[:, 0:21].copy()
            return flipped_skeleton
        else:
            return skeleton

class Scale(object):
    """
    Scale skeleton sequences by applying random scaling factors.
    
    Args:
        scale_range (tuple): Range of scaling factors (min, max).
    """
    
    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, skeleton):
        T = skeleton.shape[0]
        scales = np.random.uniform(*self.scale_range, size=T)
        scaled_skeleton = skeleton * scales[:, np.newaxis, np.newaxis]
        return scaled_skeleton

class TemporalRescale(object):
    """
    Temporally rescale video by resampling frames.
    
    Args:
        temp_scaling (float): Temporal scaling factor. Video length is scaled 
            between [1 - temp_scaling, 1 + temp_scaling].
    """
    
    def __init__(self, temp_scaling=0.2) -> None:
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))

        start_idx = 0 if random.uniform(0,1) > 0.5 else 1
        index_ = list(range(start_idx, new_len, 2))
        index_rgb = [index[num] for num in index_]
        return clip[index_rgb]

class TemporalRescale_test(object):
    def __call__(self, clip):
        # clip shape: T X N X 2
        vid_len = len(clip)
        new_len = vid_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        index = [i for i in range(new_len)]
        for i in range(vid_len, new_len):
            index[i] = index[vid_len-1]
        index_rgb = index[::2]
        return clip[index_rgb]
