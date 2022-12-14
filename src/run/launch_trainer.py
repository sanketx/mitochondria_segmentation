"""Launch a Trainer with the specified configuration file
Specify yaml overrides as additional cmd line parameters
Example - data_module.split_id:4

Usage:
  launch_trainer.py <exp_dir> [<overrides>...]
  launch_trainer.py (-h | --help)

Options:
  -h --help         Show this screen
"""

import os
import yaml
from .. import config
from docopt import docopt
from .config_parser import load_config, get_objects


def setup_artefact_dirs(exp_dir):
    dir_list = [
        "checkpoints",
        "csv_logs",
        "cmd_params",
        "default_params"
    ]

    for path in dir_list:
        artefact_dir = os.path.join(exp_dir, path)
        os.makedirs(artefact_dir, exist_ok=True)


def save_params(params, cmd_params, exp_dir):
    path = os.path.join(
        exp_dir, "default_params",
        f"{cmd_params['experiment']['uuid']}.yml"
    )

    with open(path, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)

    path = os.path.join(
        exp_dir, "cmd_params",
        f"{cmd_params['experiment']['uuid']}.yml"
    )

    with open(path, 'w') as fh:
        yaml.dump(cmd_params, fh, default_flow_style=False)


if __name__ == '__main__':
    args = docopt(__doc__)
    overrides = args["<overrides>"]
    exp_dir = os.path.join(config.EXP_ROOT, args["<exp_dir>"])
    params, cmd_params = load_config(exp_dir, overrides)
    
    if os.environ.get("LOCAL_RANK", 0) == 0:
        setup_artefact_dirs(exp_dir)
        save_params(params, cmd_params, exp_dir)

    data_module, model, trainer = get_objects(params, cmd_params)
    trainer.fit(model, datamodule=data_module)
