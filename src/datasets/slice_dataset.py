from .base_dataset import (
    BaseDataset,
    TrainMixin,
    ValMixin,
    TestMixin,
    PredictMixin
)

from .utils import weight


class SliceTrainDataset(TrainMixin, BaseDataset):
    def __init__(self, include_zlimits, **kwargs):
        super(SliceTrainDataset, self).__init__(**kwargs)
        self.include_zlimits = include_zlimits

    def weight(self, target, record):
        return weight(target, record, {0, 1, 3, 4}, self.include_zlimits)


class SliceValDataset(ValMixin, BaseDataset):
    def __init__(self, include_zlimits, **kwargs):
        super(SliceValDataset, self).__init__(**kwargs)
        self.include_zlimits = include_zlimits
        
    def weight(self, target, record):
        return weight(target, record, {2}, self.include_zlimits)


class SliceTestDataset(TestMixin, BaseDataset):
    def __init__(self, include_zlimits, **kwargs):
        super(SliceTestDataset, self).__init__(**kwargs)
        self.include_zlimits = include_zlimits
        
    def weight(self, target, record):
        return weight(target, record, {0, 1, 2, 3, 4}, self.include_zlimits)


class SlicePredictDataset(PredictMixin, BaseDataset):
    def __init__(self, include_zlimits, **kwargs):
        super(SlicePredictDataset, self).__init__(**kwargs)
        self.limit_depth = False

    def weight(self, target, record):
        return None  # this is ignored in the predict phase
