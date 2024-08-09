import pytorch_lightning as pl
import torch
from dataclasses import dataclass, field, asdict
import os
import json
import time
import torch.optim
import torch.nn as nn
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

class BaseModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.occupy_all_memory_flag = False
        if hasattr(self.config, "occupy_all_memory"):
            self.occupy_all_memory_flag = self.config.occupy_all_memory
        self.called_occupy_all_memory = False
        self.already_log_profile = False

    def log_important_info(self, batch):
        if self.already_log_profile:
            return
        
        self.logger.log_hyperparams({f"config.{key}":val for key, val in self.config.get_flatten_dict().items()})
        if self.trainer.global_rank == 0:
            batch_size = batch['images'].shape[0]
            prof = FlopsProfiler(self.model)
            prof.start_profile()
            try:
                self.step(batch)
            except Exception as e:
                print(e)
            profile_results = {
                "profile.flops_G/inst":prof.get_total_flops() / batch_size / (10 ** 9),
                "profile.macs_G/inst":prof.get_total_macs() / batch_size / (10 ** 9),
                "profile.parameters_M":prof.get_total_params() / (10 ** 6),
            }
            self.logger.log_hyperparams(profile_results)
            print(profile_results)
            prof.end_profile()
        self.already_log_profile = True

    def try_occupy_all_memory(self):
        if not self.occupy_all_memory_flag or self.called_occupy_all_memory:
            return
        self.called_occupy_all_memory = True
        print("***************** Start: occupying all memory *****************")
        tmp_list = []
        while True:
            try:
                tmp_list.append(torch.ones(1024, 1024, 512, dtype = torch.float32, device = self.model.device))
            except Exception as e:
                print(e)
                break
        for tensor in tmp_list:
            del tensor
        del tmp_list
        print("***************** End:   occupying all memory *****************")

    def training_step(self, batch, batch_idx):
        self.try_occupy_all_memory()
        output = self.step(batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"train.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        output = self.step(batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"val.{dataloader_idx}.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def step(self, batch):
        raise Exception()

    def log(self, *args, **kwargs):
        if self.trainer is None:
            return
        else:
            return super().log(*args, **kwargs)

    def get_world_size(self):
        if self.trainer is None:
            return 1
        else:
            return self.trainer.world_size

    def detach_avg(self, val):
        val = val.detach() / self.trainer.world_size
        torch.distributed.all_reduce(val)
        return val
    
    def detach_gather(self, val):
        tensor_list = [torch.zeros_like(val) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(tensor_list, val)
        return torch.cat(tensor_list, dim = 0)
    
    def sync_dict(self, inp):
        out = {key:self.detach_avg(val) for key, val in inp.items()}
        return out
    
    def gather_dict(self, inp):
        out = {key:self.detach_gather(val) for key, val in inp.items()}
        return out