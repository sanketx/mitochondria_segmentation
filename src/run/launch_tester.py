"""Run evaluation on the test sets for an experiment.
Default behavior is to test all runs, but you can run tests
for a specific run by specifying the UUID

Usage:
  launch_tester.py <exp_dir> [<uuid>]
  launch_tester.py (-h | --help)

Options:
  -h --help         Show this screen
"""

import os
import yaml

import torch
import pandas as pd
import pytorch_lightning
from collections import defaultdict

from .. import config
from docopt import docopt
from .config_parser import get_data_module, get_model


def load_params(uuid, exp_dir):
    param_path = os.path.join(exp_dir, "default_params", f"{uuid}.yml")
    cmd_param_path = os.path.join(exp_dir, "cmd_params", f"{uuid}.yml")

    with open(param_path) as fh:
        params = yaml.safe_load(fh)
        params["trainer"]["gpus"] = 1

    with open(cmd_param_path) as fh:
        cmd_params = yaml.safe_load(fh)

    return params, cmd_params


def evaluate(model, data_module):
    results = defaultdict(list)
    trainer = pytorch_lightning.Trainer(gpus=1, precision=32)
    preds = trainer.predict(model, dataloaders=data_module.test_dataloader())

    for p in preds:
        results["tomo_name"].append(p["tomo_name"][0])
        weight = {key: w.view(-1, 1) for key, w in p["weight"].items()}

        for key, w in weight.items():
            y_pred = torch.masked_select(p["y_pred"][key].view(-1, 1), w)
            y_true = torch.masked_select(p["y_true"][key].view(-1, 1), w)

            for loss_fn in model.test_loss_fns[key]:
                loss_val = loss_fn(y_pred, y_true).cpu().numpy()
                results[f"TEST{key}_{type(loss_fn).__name__}"].append(loss_val)

            for metric_fn in model.test_metric_fns[key]:
                metric_fn(y_pred, y_true)
                metric_val = metric_fn.compute().cpu().numpy()
                metric_fn.reset()
                results[f"TEST{key}_{type(metric_fn).__name__}"].append(metric_val)

    return results


def run_tests(uuid, exp_dir):
    print(f"Running tests for {uuid}")
    params, cmd_params = load_params(uuid, exp_dir)
    ckpt_path = os.path.join(exp_dir, "checkpoints", f"{uuid}.ckpt")

    model = get_model(params.pop("model"))
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    data_module = get_data_module(params.pop("data_module"), cmd_params)

    results = evaluate(model, data_module)
    result_df = pd.DataFrame.from_dict(results)
    
    result_file = os.path.join(exp_dir, "results", f"{uuid}.csv")
    result_df.to_csv(result_file, index=False)


if __name__ == '__main__':
    args = docopt(__doc__)
    uuid = args["<uuid>"]
    exp_dir = os.path.join(config.EXP_ROOT, args["<exp_dir>"])
    
    results_path = os.path.join(exp_dir, "results")
    os.makedirs(results_path, exist_ok=True)

    if uuid is not None:
        run_tests(uuid, exp_dir)

    else:
        uuid_list = [file_name.split(".")[0] for file_name in 
                     os.listdir(os.path.join(exp_dir, "checkpoints"))]

        for uuid in uuid_list:
            run_tests(uuid, exp_dir)
