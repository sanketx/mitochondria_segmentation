import torch
from torchmetrics import Metric


class DiceMetric(Metric):
    def __init__(self, threshold):
        super().__init__()
        self.thresh = threshold
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.tensor(1e-7), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.where(y_pred < self.thresh, 0.0, 1.0)

        self.intersection += torch.sum(y_true * y_pred)
        self.denom += torch.sum(y_true) + torch.sum(y_pred)

    def compute(self):
        return 2 * self.intersection / self.denom
