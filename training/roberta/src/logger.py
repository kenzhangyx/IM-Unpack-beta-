from pdb import run
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
import os
# import wandb
import mlflow
import subprocess
import time
import threading
import socket
from contextlib import closing
from argparse import Namespace

class PrivateLogger(Logger):
    def __init__(self, config):

        self.log_dir_parh = config.log_dir_parh
        self.run_name = config.run_name
        self.project = config.project if not config.trial else "trials"
        self.output_dir = config.output_dir

        self.create_logger()

    @rank_zero_only
    def create_logger(self):
        self.loggers = [CSVLogger(save_dir = self.log_dir_parh, name = "csv_logs")]

        mlflow.set_tracking_uri(f"http://192.168.21.14:10888")
        mlflow.set_experiment(self.project)
        mlflow.pytorch.autolog()
        mlflow.start_run(run_name = self.run_name)
        current_run = mlflow.active_run()
        self.experiment_id = current_run.info.experiment_id
        self.run_id = current_run.info.run_id

        logger = MLFlowLogger(
            experiment_name = mlflow.get_experiment(self.experiment_id).name,
            tracking_uri = mlflow.get_tracking_uri(),
            run_id = self.run_id,
            save_dir = os.path.join(self.output_dir, "mlruns"),
        )

        self.loggers.append(logger)

    @property
    def name(self):
        return "PrivateLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        if self.loggers is not None:
            for logger in self.loggers:
                r = logger.log_hyperparams(params)
            return r

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.loggers is not None:
            for logger in self.loggers:
                r = logger.log_metrics(metrics, step)
            return r

    @rank_zero_only
    def finalize(self, status):
        if self.loggers is not None:
            for logger in self.loggers:
                r = logger.finalize(status)
            mlflow.end_run()
            return r