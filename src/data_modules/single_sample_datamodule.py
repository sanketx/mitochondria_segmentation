from .base_datamodule import BaseDataModule


class SingleSampleDataModule(BaseDataModule):
    def __init__(self, sample, split_id, split_type, **kwargs):
        super(SingleSampleDataModule, self).__init__(**kwargs)
        self.sample = sample
        self.split_id = split_id
        self.split_type = split_type

    def train_df(self):
        return self.record_df[
            (self.record_df[self.split_type] != self.split_id) &
            (self.record_df["sample"] == self.sample)
        ]

    def val_df(self):
        return self.record_df[
            (self.record_df[self.split_type] != self.split_id) &
            (self.record_df["sample"] == self.sample)
        ]

    def test_df(self):
        return self.record_df[
            (self.record_df[self.split_type] == self.split_id) &
            (self.record_df["sample"] == self.sample)
        ]

    def predict_df(self):
        return self.records_df
