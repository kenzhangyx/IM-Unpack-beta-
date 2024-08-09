
import shutil
import os
import pytorch_lightning as pl
from src.args import import_config, import_from_string
import argparse
import datetime
import logging
import copy
import sys
import json
import torch
import random
import time
import logging
from src.logger import PrivateLogger
from src.args import import_from_string

import mlflow

def main(config):

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval = "step")]
    if config.save_last_n_checkpoints > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last = True,
                save_top_k = config.save_last_n_checkpoints,
                dirpath = config.save_dir_path,
                monitor = "step",
                mode = "max",
                filename = "{epoch:05d}-{step:08d}",
                save_on_train_epoch_end = False,
                every_n_epochs = 0 if config.save_last_n_checkpoints == 0 else None
            )
        )

    if hasattr(config, "callbacks"):
        for callback in config.callbacks:
            callbacks.append(import_from_string(callback.type)(**callback.args.to_dict()))
    
    trainer_logger = PrivateLogger(config)

    trainer = pl.Trainer(
        **config.trainer.to_dict(),
        callbacks = callbacks,
        enable_checkpointing = (config.save_last_n_checkpoints > 0),
        default_root_dir = config.save_dir_path if config.save_last_n_checkpoints > 0 else None,
        accelerator = 'gpu',
        logger = trainer_logger
    )

    if not os.path.exists(config.save_dir_path) and trainer.global_rank == 0:
        os.makedirs(config.save_dir_path)

    if trainer.global_rank == 0:
        print(config)

    torch.manual_seed(config.seed * (trainer.global_rank + 1))

    print(f"*********** initializing data module ***********")
    data = import_from_string(config.data.pl_module)(config)
        
    print(f"*********** initializing model module ***********")
    model = import_from_string(config.model.pl_module)(config, data_module = data)
    
    print(f"*********** seting up data module ***********")
    data.setup()
        
    if trainer.global_rank == 0:
        print(callbacks)
        print(trainer)
        print(data)
        print(model)

    if config.mode == "train":
        print(f"*********** start training ***********")
        trainer.fit(model = model, datamodule = data, ckpt_path = possible_ckpt_path)
    elif config.mode == "val":
        print(f"*********** start validation ***********")
        trainer.validate(model = model, datamodule = data, ckpt_path = possible_ckpt_path)
    elif config.mode == "test":
        print(f"*********** start testing ***********")
        trainer.test(model = model, datamodule = data, ckpt_path = possible_ckpt_path)
    else:
        raise Exception()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument('--mode', type = str, default = "train")
    parser.add_argument('--trial', action = 'store_true')
    args = parser.parse_args()

    print(f"args: {args}")

    assert args.mode in ["train", "val", "test"]

    config = import_config(args.config)
    config.run_name = args.mode + "-" + args.config.replace(os.sep, "-")
    if config.trainer.devices == -1:
        config.trainer.devices = torch.cuda.device_count()
    config.save_dir_path = os.path.join(config.output_dir, config.project, args.config.replace(os.sep, "-"), "ckpts")
    config.log_dir_parh = os.path.join(config.output_dir, config.project, args.config.replace(os.sep, "-"), "logs")
    config.misc_dir_path = os.path.join(config.output_dir, config.project, args.config.replace(os.sep, "-"), "misc")

    os.makedirs(config.save_dir_path, exist_ok = True)
    os.makedirs(config.log_dir_parh, exist_ok = True)
    os.makedirs(config.misc_dir_path, exist_ok = True)
    
    config.config_path = args.config
    config.mode = args.mode
    config.trial = args.trial
    
    if config.trial:
        config.trainer.devices = 1
        config.trainer.num_nodes = 1
        config.data.batch_size = min(2, config.data.batch_size)

    if not config.trial and config.mode in ["val", "test"]:
        config.data.batch_size = config.data.batch_size * config.trainer.num_nodes
        config.trainer.num_nodes = 1

    if config.mode == "train" and os.path.exists(os.path.join(config.save_dir_path, "last.ckpt")):
        config.seed = config.seed * random.randrange(10000)
        print(f"new seed: {config.seed}")

    print(f"*********** finding potential resume checkpoints ***********")
    possible_ckpt_path = os.path.join(config.save_dir_path, "last.ckpt")
    if os.path.exists(possible_ckpt_path):
        pass
    elif hasattr(config, "resume_ckpt_path"):
        possible_ckpt_path = config.resume_ckpt_path
    else:
        possible_ckpt_path = None

    if config.mode in ["val", "test"] and config.evaluate_ckpt is not None:
        possible_ckpt_path = config.evaluate_ckpt

    if possible_ckpt_path is not None:
        print(f"Resuming from checkpoint to {possible_ckpt_path}")

    config.possible_ckpt_path = possible_ckpt_path

    main(config)
