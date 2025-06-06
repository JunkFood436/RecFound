import os
import pandas as pd
import sys
import torch
from utils.common_utils import TASK2ID, ID2TASK, EMBTASKLIST, GENTASKLIST, TASK_NUMS
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union

task_slope_list = None


class MFTLossStatus:
    def __init__(self):
        super(MFTLossStatus, self).__init__()


class CoBaStatus(MFTLossStatus):
    def __init__(
        self,
        coba_warmup_steps=100,
        coba_history_length=200,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
        gradient_accumulation_steps=128,
        valid_dataloader=None,
    ):

        super(CoBaStatus, self).__init__()
        self.coba_warmup_steps = coba_warmup_steps
        self.coba_history_length = coba_history_length
        self.coba_tau = coba_tau
        self.coba_update_interval = coba_update_interval
        self.coba_sample_valid_num = coba_sample_valid_num
        self.valid_dataloader = valid_dataloader
        self.valid_dataloader_length = len(valid_dataloader)
        self.valid_iterator = iter(valid_dataloader)
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK))
        self.history_task_valid_loss = None
        self.per_task_slope_list = None
        self.emb_total_slope_list = torch.tensor([], dtype=torch.float64)
        self.gen_total_slope_list = torch.tensor([], dtype=torch.float64)
        self.minimum_weight = 1 / (len(ID2TASK) * 2)
        self.valid_task_loss_begining = torch.ones(len(ID2TASK), dtype=torch.float64)
        self.log_per_task_weight = torch.ones(len(ID2TASK)) / len(ID2TASK)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.emb_min_weight = 1 / (len(EMBTASKLIST) * 2)
        self.gen_min_weight = 1 / (len(GENTASKLIST) * 2)

    def coba_evaluate(self, model, v_batch, per_task_weight=None):
        model.eval()
        with torch.no_grad():
            valid_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
            gen_task_id = torch.tensor(v_batch['gen_task']).to(model.device)
            emb_task_id = torch.tensor(v_batch['emb_task']).to(model.device)
            gen_loss = model(generative=v_batch["generative"], gen_task=gen_task_id,loss_weight=per_task_weight).loss
            emb_loss = model(query=v_batch["query"], passage=v_batch["passage"], emb_task=emb_task_id,loss_weight=per_task_weight).loss
            valid_task_loss[gen_task_id] += gen_loss
            valid_task_loss[emb_task_id] += emb_loss * torch.log2(torch.tensor(self.gradient_accumulation_steps))

            task_exist = (valid_task_loss != 0.0).float()
            torch.distributed.all_reduce(valid_task_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(task_exist, op=torch.distributed.ReduceOp.SUM)
            valid_task_loss /= task_exist.clamp_(1.0)
            valid_task_loss /= self.valid_task_loss_begining
        model.train()
        return valid_task_loss

    def compute_per_task_weight(self, completed_steps=None):
        task_num = len(ID2TASK)
        emb_num = len(EMBTASKLIST)
        gen_num = len(GENTASKLIST)
        device = self.history_task_valid_loss.device

        # ================ Public Parameters ================
        start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1, device=device)
        
        task_slope_fitting = torch.zeros(task_num, dtype=torch.float64, device=device)
        for i in range(task_num):
            per_task_history = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(history_steps, per_task_history, "slope")

        if completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
        else:
            self.per_task_slope_list = torch.cat(
                (self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), 
                dim=-1
            )
        task_slope_list = self.per_task_slope_list

        # ================ Group Parameters ================
        # Embedding Loss
        emb_history_loss, _ = torch.max(self.history_task_valid_loss[:emb_num, -len(history_steps):], dim=0)
        emb_total_slope = self.fit_window_slope(history_steps, emb_history_loss, "slope")
        
        # Generative Loss
        gen_history_loss, _ = torch.max(self.history_task_valid_loss[emb_num:, -len(history_steps):], dim=0)
        gen_total_slope = self.fit_window_slope(history_steps, gen_history_loss, "slope")

        
        self.emb_total_slope_list = torch.cat((self.emb_total_slope_list, emb_total_slope.unsqueeze(0)))
        self.gen_total_slope_list = torch.cat((self.gen_total_slope_list, gen_total_slope.unsqueeze(0)))

        def compute_divergence(slope_history):
            if len(slope_history) < 1:
                return torch.tensor(1.0, device=device)
            
            normalized = -len(slope_history) * slope_history / slope_history.abs().sum()
            divergence = F.softmax(normalized * self.coba_tau, dim=-1)[-1] * len(slope_history)
            return torch.min(divergence, torch.tensor(1.0, device=device))

        emb_divergence = compute_divergence(self.emb_total_slope_list)
        gen_divergence = compute_divergence(self.gen_total_slope_list)

        def process_group(indices, group_size, divergence):
            group_slope = task_slope_fitting[indices]
            group_history = self.per_task_slope_list[indices, start_step:]
            
            normalized_rcs = group_size * group_slope / group_slope.abs().sum()
            rcs = F.softmax(normalized_rcs, dim=0)
            
            reversed_norm = -group_history.shape[1] * group_history / group_history.abs().sum(dim=-1, keepdim=True)
            current_slope = reversed_norm.T.reshape(-1)[-group_size:]
            acs = F.softmax(current_slope, dim=0)
            
            weight_logits = divergence * rcs + (1 - divergence) * acs

            return F.softmax(weight_logits * group_size, dim=0)

        emb_weights = process_group(slice(0, emb_num), emb_num, emb_divergence)
        gen_weights = process_group(slice(emb_num, task_num), gen_num, gen_divergence)
        
        emb_ratio = torch.tensor(TASK_NUMS[:emb_num],dtype=emb_weights.dtype)
        emb_ratio = emb_ratio / emb_ratio.mean()
        emb_weights = (emb_weights * emb_ratio) / torch.norm(emb_weights * emb_ratio, p=1)

        gen_ratio = torch.tensor(TASK_NUMS[emb_num:],dtype=gen_weights.dtype)
        gen_ratio = gen_ratio / gen_ratio.mean()
        gen_weights = (gen_weights * gen_ratio) / torch.norm(gen_weights * gen_ratio, p=1)


        per_task_weight = torch.cat([emb_weights, gen_weights])

        def adjust_weights(weights, group_size):
            min_weight = 1 / (group_size * 5)
            under_min = weights < min_weight
            if under_min.any():
                adjust_factor = 1 - min_weight * group_size
                weights = weights * adjust_factor + min_weight
            return weights

        per_task_weight = torch.cat([
            adjust_weights(per_task_weight[:emb_num], emb_num),
            adjust_weights(per_task_weight[emb_num:], gen_num)
        ])

        return per_task_weight
        
    def fit_window_slope(self, x, y, type="slope"):

        y = y[y != 0]
        x = x[:len(y)]
        
        nonzero_index = torch.squeeze(torch.nonzero(y), dim=1)
        y = torch.index_select(y, 0, nonzero_index)
        x = torch.index_select(x, 0, nonzero_index)

        ws = torch.flip(1 ** torch.arange(len(y)), dims=[0])
        ws = ws.double()

        if len(y) >= 2:
            if type == "slope":
                X = torch.stack((x, torch.ones_like(x, dtype=torch.float64))).T
                X = X.double()
            else:
                X = torch.stack((x ** 2, x, torch.ones_like(x, dtype=torch.float64))).T

            # implementation for numpy
            # X_np = X.T @ (ws[:, None] * X)
            # Y_np = X.T @ (ws * y)
            # w = torch.from_numpy(np.linalg.solve(X_np.numpy(), Y_np.numpy()))

            # implementation for torch
            w = torch.linalg.solve(X.T @ (ws[:, None] * X), X.T @ (ws * y))

            result = w[0]
        else:
            result = 0.0

        return result

    def sample_valid_batch(self, model, completed_steps):
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK), dtype=torch.float64)
        self.valid_task_loss_exist = torch.zeros(len(ID2TASK), dtype=torch.float64)
        for i in range(self.coba_sample_valid_num):
            if (
                self.coba_sample_valid_num * completed_steps // self.coba_update_interval + i
            ) % self.valid_dataloader_length == 0:
                self.valid_iterator = iter(self.valid_dataloader)
                v_batch = next(self.valid_iterator)
            else:
                v_batch = next(self.valid_iterator)
            valid_task_loss = self.coba_evaluate(model, v_batch, self.log_per_task_weight)
            self.valid_task_loss_exist += (valid_task_loss == 0).detach().cpu().float()
            self.valid_task_loss_accumulated += valid_task_loss.detach().cpu().double()


        self.valid_task_loss_accumulated /= self.valid_task_loss_exist.clamp_(1.0)

        # self.valid_task_loss_accumulated /= self.coba_sample_valid_num
        if self.history_task_valid_loss is None and completed_steps >= 1:
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        elif self.history_task_valid_loss is not None:
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )
