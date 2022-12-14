import os
import math
import h5py
import numpy as np
from torch.utils.data import Dataset

from .. import config
from .utils import pad_and_expand, truncate
from ..transforms.transforms import transform_volumes


class BaseDataset(Dataset):
    def __init__(self, records, targets, augment, rf=1):
        self.records = records
        self.targets = targets
        self.augment = augment
        self.rf = rf  # replication factor
        
        self.num_records = len(records)

    def __len__(self):
        if self.rf == 1:
            return self.num_records

        # round up the new size to nearest multiple of 4
        return math.ceil(self.num_records * self.rf / 4.0) * 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        record = self.records[idx % self.num_records].copy()
        self.load_tomogram(record)  # modify record in-place
        
        if self.limit_depth:
            record = truncate(record)

        X = self.get_input(record)
        Y = self.get_outputs(record)
        W = self.get_weights(record)
        
        if self.augment:
            X, Y = transform_volumes(X, Y)

        X, Y, W = self.post_process(X, Y, W, record)
        return pad_and_expand(X, Y, W, record)

    def load_tomogram(self, record):
        record["data_keys"] = set(self.targets) | {"data"}

        tomo_path = os.path.join(
            config.TRAIN_TOMO_DIR,
            record["sample"],
            record["tomo_name"]
        )

        with h5py.File(tomo_path) as fh:
            for key in record["data_keys"]:
                record[key] = fh[key][()].astype(np.float32)

        record["shape"] = record["data"].shape

    def get_input(self, record):
        return record["data"]

    def get_outputs(self, record):
        return {t: record[t] for t in self.targets}

    def get_weights(self, record):
        return {t: self.weight(t, record) for t in self.targets}

    # Override for distance transform generation
    def post_process(self, X, Y, W, record):
        return X, Y, W


class TrainMixin:
    def __init__(self, **kwargs):
        super(TrainMixin, self).__init__(**kwargs)
        self.limit_depth = True


class ValMixin:
    def __init__(self, **kwargs):
        super(ValMixin, self).__init__(**kwargs)
        self.limit_depth = True
        self.augment = False
        self.rf = 1


class TestMixin:
    def __init__(self, **kwargs):
        super(TestMixin, self).__init__(**kwargs)
        self.limit_depth = False
        self.augment = False
        self.rf = 1


class PredictMixin:
    def __init__(self, **kwargs):
        super(PredictMixin, self).__init__(**kwargs)
        self.limit_depth = False
        self.augment = False
        self.rf = 1


if __name__ == '__main__':
    pass
