import pytorch_lightning as pl
import torch
import os
import json
import time
from transformers import RobertaPreTrainedModel, AutoConfig, PretrainedConfig
import torch.nn as nn
from .metrics import Loss, Accuracy
from src.base import BaseModelModule
from src.utilities.utils import get_scheduler, get_optimizer
from transformers.trainer_pt_utils import get_parameter_names
from src.args import import_from_string

class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gelu = nn.GELU()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        self.bias = self.decoder.bias

class RobertaForMLM(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.roberta = import_from_string(config.encoder_type)(config)
        self.lm_head = LMHead(config)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.post_init()

    def forward(self, labels, **kwargs):
        sequence_output, auxiliary, extra_outputs = self.roberta(**kwargs)

        if "masked_token_indices" in kwargs:
            masked_token_indices = kwargs["masked_token_indices"]
            batch_indices = torch.arange(sequence_output.shape[0], device = sequence_output.device)[:, None]
            sequence_output = sequence_output[batch_indices, masked_token_indices, :]
            labels = labels[batch_indices, masked_token_indices]

        logits = self.lm_head(sequence_output)
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        mlm_loss, _ = self.loss_fct(logits, labels)
        mlm_accu, mlm_count = self.accu_fct(logits, labels)

        if len(auxiliary) == 0:
            output = {
                "loss":mlm_loss, 
                "mlm_loss":mlm_loss, 
                "mlm_accu":mlm_accu, 
                "mlm_count":mlm_count
            }
            output.update(extra_outputs)
        else:
            auxiliary_loss = []
            for key, val in auxiliary.items():
                key = key.split("-")[0]
                auxiliary_loss.append(getattr(self.config, f"auxiliary_{key}") * val)
            auxiliary_loss = sum(auxiliary_loss) / len(auxiliary_loss)

            output = {
                "loss":mlm_loss + auxiliary_loss, 
                "auxiliary_loss":auxiliary_loss, 
                "mlm_loss":mlm_loss, 
                "mlm_accu":mlm_accu, 
                "mlm_count":mlm_count
            }
            output.update(extra_outputs)
            output.update(auxiliary)

        return output

    def resize_position_embeddings(self, target_length):
        gathered = []
        for module in self.roberta.modules():
            if hasattr(module, "resize_position_embeddings"):
                gathered.append(module)
        for module in gathered:
            module.resize_position_embeddings(target_length, self._init_weights)

    def resize_token_type_embeddings(self, target_length):
        gathered = []
        for module in self.roberta.modules():
            if hasattr(module, "resize_token_type_embeddings"):
                gathered.append(module)
        for module in gathered:
            module.resize_token_type_embeddings(target_length, self._init_weights)

class MLMModelModule(BaseModelModule):
    def __init__(self, config, data_module):
        super().__init__(config)

        if hasattr(self.config.model, "pretrained_config"):
            self.model_config = AutoConfig.from_pretrained(self.config.model.pretrained_config)
        else:
            self.model_config = PretrainedConfig()
        for key, val in self.config.model.to_dict().items():
            setattr(self.model_config, key, val)
        print(self.model_config)

        self.model = RobertaForMLM(self.model_config)
        self.tokenizer = data_module.tokenizer

        if hasattr(self.config.model, "load_checkpoint"):
            states = torch.load(self.config.model.load_checkpoint)
            missing_keys, unexpected_keys = self.load_state_dict(states["state_dict"], strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")

    def step(self, batch):
        output = self.model(**batch)
        return output

    def configure_optimizers(self):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        params_decay = [p for n, p in self.named_parameters() if any(nd in n for nd in decay_parameters)]
        params_nodecay = [p for n, p in self.named_parameters() if not any(nd in n for nd in decay_parameters)]

        optim_groups = [
            {"params": params_decay, "weight_decay": self.config.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = get_optimizer(optim_groups = optim_groups, **self.config.optimizer.to_dict())

        max_steps = self.trainer.max_steps
        if max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
            print(f"Inferring max_steps: {max_steps}")

        scheduler = get_scheduler(
            self.config.optimizer.lr_scheduler_type,
            optimizer,
            num_warmup_steps = self.config.optimizer.warmup_steps,
            num_training_steps = max_steps,
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "loss",
                }
            ],
        )