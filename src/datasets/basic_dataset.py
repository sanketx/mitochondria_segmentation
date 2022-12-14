from .base_dataset import (
    BaseDataset,
    TrainMixin,
    ValMixin,
    TestMixin,
    PredictMixin
)

from .utils import weight


class BasicDataset(BaseDataset):
    def __init__(self, include_zlimits, **kwargs):
        super(BasicDataset, self).__init__(**kwargs)
        self.include_zlimits = include_zlimits

    def weight(self, target, record):
        return weight(target, record, {0, 1, 2, 3, 4}, self.include_zlimits)


class BasicTrainDataset(TrainMixin, BasicDataset):
    pass


class BasicValDataset(ValMixin, BasicDataset):
    pass


class BasicTestDataset(TestMixin, BasicDataset):
    pass


class BasicPredictDataset(PredictMixin, BasicDataset):
    pass
