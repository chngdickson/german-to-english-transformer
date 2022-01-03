import numpy as np
from random import shuffle
from torch.utils.data import Sampler

class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source, tokenizer,
                bucket_boundaries, batch_size=64,):
        self.ind_n_len = [(i, len(tokenizer(s[1]))) for i, s in enumerate(data_source)]
        
        self.data_source = data_source
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
        
    def __iter__(self):
        shuffle(self.ind_n_len)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(self.ind_n_len), self.batch_size * 100):
            pooled_indices.extend(sorted(self.ind_n_len[i:i + self.batch_size * 100], key=lambda x: x[1])) # as it was stored in an array
        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]
        
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id