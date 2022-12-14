import torch
from torch import nn
from pytorch_lightning import LightningModule


class BaseModule(LightningModule):
    def __init__(self, losses, metrics, lr=1e-3):
        super(BaseModule, self).__init__()
        self.lr = lr
        self.build()
        self.compile(losses, metrics)

    def build(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def compile(self, losses, metrics):
        self.train_loss_fns = {
            k: [fn(**params) for fn, params in v]
            for k, v in losses.items()
        }
        
        self.val_loss_fns = {
            k: [fn(**params) for fn, params in v]
            for k, v in losses.items()
        }

        self.test_loss_fns = {
            k: [fn(**params) for fn, params in v]
            for k, v in losses.items()
        }

        self.train_metric_fns = nn.ModuleDict({
            k: nn.ModuleList([fn(**params) for fn, params in v])
            for k, v in metrics.items()
        })

        self.val_metric_fns = nn.ModuleDict({
            k: nn.ModuleList([fn(**params) for fn, params in v])
            for k, v in metrics.items()
        })
        
        self.test_metric_fns = nn.ModuleDict({
            k: nn.ModuleList([fn(**params) for fn, params in v])
            for k, v in metrics.items()
        })
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def masked_predict(self, batch):
        x = batch['X']
        y_true = batch['Y']  # a dict of tensors
        weight = batch['W']  # a dict of tensors

        weight = {key: w.view(-1, 1).detach() for key, w in weight.items()}
        y_pred = self(x)  # a dict of tensors

        for key, w in weight.items():
            y_pred[key] = torch.masked_select(y_pred[key].view(-1, 1), w)
            y_true[key] = torch.masked_select(y_true[key].view(-1, 1), w)

        return y_pred, y_true

    def step(self, batch, loss_fns, prefix):
        y_pred, y_true = self.masked_predict(batch)
        losses = []

        for key in loss_fns:
            yp = y_pred[key]
            yt = y_true[key]

            for loss_fn in loss_fns[key]:
                loss_val = loss_fn(yp, yt)
                losses.append(loss_val)

                self.log(
                    f"{prefix}{key}_{type(loss_fn).__name__}",
                    loss_val,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )

        return {
            "loss": sum(losses),
            "y_pred": {key: yp.detach() for key, yp in y_pred.items()},
            "y_true": y_true,
        }

    def step_end(self, outputs, metric_fns, prefix):
        y_pred, y_true = outputs["y_pred"], outputs["y_true"]

        for key in metric_fns:
            yp = y_pred[key]
            yt = y_true[key]

            for metric_fn in metric_fns[key]:
                metric_fn(yp, yt)

                self.log(
                    f"{prefix}{key}_{type(metric_fn).__name__}",
                    metric_fn,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )

    def training_step(self, batch, batch_idx):
        return self.step(batch, self.train_loss_fns, "TRAIN")

    def training_step_end(self, outputs):
        self.step_end(outputs, self.train_metric_fns, "TRAIN")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, self.val_loss_fns, "VAL")

    def validation_step_end(self, outputs):
        self.step_end(outputs, self.val_metric_fns, "VAL")

    def test_step(self, batch, batch_idx):
        return self.step(batch, self.test_loss_fns, "TEST")

    def test_step_end(self, outputs):
        self.step_end(outputs, self.test_metric_fns, "TEST")

    def predict_step(self, batch, batch_idx):
        return {
            "y_pred": self(batch['X']),
            "y_true": batch['Y'],
            "weight": batch['W'],
            "tomo_name": batch["tomo_name"]
        }
