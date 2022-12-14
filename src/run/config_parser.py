import os
import yaml
import copy
import uuid
from pprint import pprint

import pytorch_lightning
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .. import datasets
from .. import data_modules
from .. import models
from ..models.blocks import OutputLayer


def get_overrides(param_string):
    idx = param_string.index(":")
    key_list = param_string[:idx].split('.')
    value = param_string[idx + 1:]
    return key_list, value


def update_params(root, key_list, value):
    for key in key_list[:-1]:
        if key not in root:
            root[key] = {}

        root = root[key]

    try:
        root[key_list[-1]] = eval(value)
    except NameError:
        root[key_list[-1]] = value


def load_config(exp_dir, overrides=[]):
    cmd_params = {
        "experiment": {
            "exp_dir": exp_dir,
            "uuid": str(uuid.uuid1())
        }
    }

    config_file = os.path.join(exp_dir, "params.yml")

    with open(config_file) as fh:
        params = yaml.safe_load(fh)
    
    for o in overrides:
        key_list, value = get_overrides(o)
        update_params(params, key_list, value)
        update_params(cmd_params, key_list, value)

    validate_config(params)
    return params, cmd_params


def get_objects(default_params, cmd_params):
    params = copy.deepcopy(default_params)

    data_module = get_data_module(params.pop("data_module"), cmd_params)
    model = get_model(params.pop("model"))
    trainer = get_trainer(params.pop("trainer"), cmd_params)

    return data_module, model, trainer


def validate_config(params):
    """
    Validate the experiment parameters to avoid errors in initialization
    """
    pass


def get_data_module(params, cmd_params):
    dm_class = getattr(data_modules, params.pop("name"))
    dataset_params = params.pop("dataset_params")
    dataloader_params = params.pop("dataloader_params")
    dataset_class = params.pop("dataset_class")

    exp_dir = cmd_params["experiment"]["exp_dir"]
    split_file = os.path.join(exp_dir, "splits.csv")

    for key, value in dataset_class.items():
        dataset_class[key] = getattr(datasets, value)

    return dm_class(
        split_file=split_file,
        dataset_class=dataset_class,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
        **params
    )


def get_model(params):
    model_class = getattr(models, params.pop("name"))
    outputs = params.pop("outputs")
    losses, metrics, activations = {}, {}, {}

    for key, value in outputs.items():
        losses[key] = [
            # Loss function class, constructor params
            (getattr(models.losses, fn_params.pop("name")), fn_params)
            for fn_params in value.pop("losses")
        ]

        metrics[key] = [
            # Loss function class, constructor params
            (getattr(models.metrics, fn_params.pop("name")), fn_params)
            for fn_params in value.pop("metrics")
        ]

        if (act_params := value.pop("activation")) is not None:
            act_class = getattr(models.blocks, act_params.pop("name"))
            activations[key] = act_class(**act_params)

        else:
            activations[key] = None

    return model_class(
        output_layer=OutputLayer(activations),
        losses=losses,
        metrics=metrics,
        **params,
    )    


def get_trainer(params, cmd_params):
    exp_params = cmd_params["experiment"]
    wandb_params = params.pop("wandb_params")

    callback_params = params.pop("callbacks")
    monitor = callback_params.pop("monitor")
    mode = callback_params.pop("mode")
    es_params = callback_params.pop("early_stopping")

    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_params["exp_dir"], "checkpoints"),
        filename=exp_params["uuid"],
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        check_on_train_epoch_end=False,
        **es_params,
    )

    csv_logger = CSVLogger(
        save_dir=os.path.join(exp_params["exp_dir"], "csv_logs"),
        version=exp_params["uuid"],
        name="",
        flush_logs_every_n_steps=1
    )

    wandb_logger = WandbLogger(
        id=exp_params["uuid"],
        save_dir=exp_params["exp_dir"],
        project="mito_methods",
        **wandb_params,
    )

    if os.environ.get("LOCAL_RANK", 0) == 0:
        wandb_logger.experiment.config.update(cmd_params)

    return pytorch_lightning.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        logger=[csv_logger, wandb_logger],
        callbacks=[model_checkpoint, early_stopping],
        **params,
    )
