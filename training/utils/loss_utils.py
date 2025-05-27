import os
import pandas as pd
import sys
import torch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union

TASK_NUMS = [9095, 4090, 8837, 579, 1985, 2040, 1994, 2012, 400, 1239, 1499, 1972, 2001]

log_embdf = []
log_gendf = []
task_slope_list = None
log_embrcs = []
log_embacs = []
log_genrcs = []
log_genacs = []


def get_task_mask(task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1

    return task_mask


def get_task_loss(task_losses, task_id):  # TODO
    # fix task order
    task_loss_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # count task samples
    task_num_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i in range(len(task_id)):
        task_num_per_batch[task_id[i][0]] += 1
        task_loss_per_batch[task_id[i][0]] = task_losses[task_id[i][0]]

    return task_loss_per_batch, task_num_per_batch


def loss_func_mft(outputs, labels, task_mask, task_id, weighted_loss_mode, loss_mask=None, task_weights=None):
    """
    loss function for MFT loss
    :param outputs:
    :param labels:
    :param task_mask:
    :param task_id:
    :param weighted_loss_mode:
    :param loss_mask:
    :return:
    """
    # task_id shape: [[1], [2], [4], [3], ..., [1]]
    weighted = weighted_loss_mode
    lm_logits = outputs["logits"]
    labels = labels.to(device=lm_logits.device)
    task_mask = task_mask.to(device=lm_logits.device)
    task_id = task_id.to(device=lm_logits.device)
    shift_logits = lm_logits.contiguous()
    labels = labels.contiguous()
    if task_weights is None:
        task_weights = torch.ones(len(ID2TASK)).to(device=lm_logits.device) / len(ID2TASK)

    bsz, seq_len = labels.shape
    # loss_mask = None
    if loss_mask is None:
        ineffective_tokens_per_sample = (labels == -100).sum(dim=1)
        effective_tokens_per_sample = -(ineffective_tokens_per_sample - seq_len)
        effective_tokens = bsz * seq_len - ineffective_tokens_per_sample.sum()
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    else:
        loss_mask = loss_mask.to(device=lm_logits.device)
        loss_fct = CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))  # [B * L, 1]
    losses = losses.contiguous().view(bsz, -1)
    token_losses = (
        losses.clone().detach().float() if loss_mask is None else losses.clone().detach().float() * loss_mask
    )  # [B, L]
    task_mask_trans = torch.transpose(task_mask, 0, 1)
    unique_id = torch.unique(task_id)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4" or weighted_loss_mode == "coba":
        loss = 0.0
        weights_sum = 0.0
        for i, w in enumerate(unique_id):
            row_idx = torch.squeeze(task_id) == w.item()
            task_weight = float(task_weights[w.item()])
            weights_sum += task_weight
            if weighted_loss_mode == "case3" or weighted_loss_mode == "coba":
                if loss_mask is None:
                    loss += (
                        torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx]) * task_weight
                    )
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :]) * task_weight
            elif weighted_loss_mode == "case4":
                if loss_mask is None:
                    loss += (
                        torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx])
                        * task_weight
                    )
                else:
                    loss += (
                        torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx])
                        * task_weight
                    )

        # loss /= len(unique_id)
        loss /= weights_sum

    elif weighted_loss_mode == "case2":
        if loss_mask is None:
            loss = torch.mean(torch.sum(losses, dim=1) / effective_tokens_per_sample)
        else:
            loss = torch.mean(torch.sum(losses * loss_mask, dim=1) / torch.sum(loss_mask, dim=1))
    elif weighted_loss_mode == "case1":
        # flatten losses & loss_mask tensor
        if loss_mask is None:
            # losses = losses.view(-1)
            loss = torch.sum(losses.view(-1)) / effective_tokens
        else:
            # loss_mask = loss_mask.view(-1)
            # losses = losses.view(-1)
            loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.view(-1).sum()

    # fix task order
    task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i, w in enumerate(unique_id):
        row_idx = torch.squeeze(task_id) == w.item()
        if loss_mask is None:
            task_loss[w] = torch.sum(token_losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
            task_num[w] = len(effective_tokens_per_sample[row_idx])
        else:
            task_loss[w] = torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])

    return loss, task_loss, task_num


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


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
        self.gradient_accumulation_steps = 128
        self.emb_min_weight = 1 / (3 * 2)  # 0.1667
        self.gen_min_weight = 1 / (10 * 2) # 0.05

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

    # def compute_per_task_weight(self, completed_steps=None):
    #     task_num = len(ID2TASK)
    #     task_slope_fitting = torch.ones(task_num, dtype=torch.float64)
    #     start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
    #     history_steps = torch.arange(start_step, completed_steps, 1)
    #     for i in range(task_num):
    #         per_task_history_valid_loss = self.history_task_valid_loss[i][-len(history_steps):]
    #         task_slope_fitting[i] = self.fit_window_slope(
    #             history_steps, per_task_history_valid_loss, type="slope"
    #         )
    #     history_total_valid_loss, index = torch.max(self.history_task_valid_loss[:, -len(history_steps):], dim=0)
    #     total_slope_fitting = self.fit_window_slope(
    #         history_steps, history_total_valid_loss, type="slope"
    #     )
    #     if completed_steps == self.coba_warmup_steps:
    #         self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
    #         self.total_slope_list = total_slope_fitting.unsqueeze(0)
    #     else:
    #         self.per_task_slope_list = torch.cat((self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), dim=-1)
    #         self.total_slope_list =  torch.cat((self.total_slope_list, total_slope_fitting.unsqueeze(0)), dim=0)
        
    #     # Relative Convergence Score
    #     normalize_task_slope = task_num * task_slope_fitting / task_slope_fitting.abs().sum()
    #     rcs = F.softmax(normalize_task_slope, dim=-1)
        
    #     # Absolute Convergence Score
    #     history_per_task_slope_list = self.per_task_slope_list[:, start_step:]
    #     reverse_normailize_iter_slope = -len(history_per_task_slope_list[0]) * history_per_task_slope_list \
    #                                     / history_per_task_slope_list.abs().sum(dim=-1, keepdim=True)

    #     flatten_rn_iter_slope = reverse_normailize_iter_slope.T.reshape(-1)
    #     current_step_rn_slope = flatten_rn_iter_slope[-task_num:]
    #     acs = F.softmax(current_step_rn_slope, dim=-1)

    #     # Divergence Factor
    #     normalize_total_iter_slope = - len(self.total_slope_list) * self.total_slope_list \
    #                                  / self.total_slope_list.abs().sum()
    #     divergence_factor = F.softmax(normalize_total_iter_slope * self.coba_tau, dim=-1)[-1] \
    #                       * len(self.total_slope_list)
        
    #     if torch.distributed.get_rank() == 0:
    #         print("RCS:", rcs)
    #         print("ACS:", acs)
    #         print("Task Slope:", normalize_task_slope)
    #         print("DF:", divergence_factor)

    #     divergence_factor = torch.min(divergence_factor, torch.tensor(1.0))
    #     weight_logits = divergence_factor * rcs + (1 - divergence_factor) * acs
    #     per_task_weight = F.softmax(weight_logits * task_num, dim=-1)

    #     if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
    #         per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
    #         per_task_weight += self.minimum_weight

    #     return per_task_weight

    def compute_per_task_weight(self, completed_steps=None):
        task_num = len(ID2TASK)
        emb_num = 3  # Embedding任务数量
        gen_num = 10  # Generative任务数量
        device = self.history_task_valid_loss.device  # 保持设备一致

        # ================ 公共参数计算 ================
        start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1, device=device)
        
        # 计算各任务斜率
        task_slope_fitting = torch.zeros(task_num, dtype=torch.float64, device=device)
        for i in range(task_num):
            per_task_history = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(history_steps, per_task_history, "slope")

        # ================ 历史斜率记录 ================
        if completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
        else:
            self.per_task_slope_list = torch.cat(
                (self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), 
                dim=-1
            )
        task_slope_list = self.per_task_slope_list
        print(task_slope_list)

        # ================ 分组总斜率计算 ================
        # Embedding组总损失
        emb_history_loss, _ = torch.max(self.history_task_valid_loss[:emb_num, -len(history_steps):], dim=0)
        emb_total_slope = self.fit_window_slope(history_steps, emb_history_loss, "slope")
        
        # Generative组总损失
        gen_history_loss, _ = torch.max(self.history_task_valid_loss[emb_num:, -len(history_steps):], dim=0)
        gen_total_slope = self.fit_window_slope(history_steps, gen_history_loss, "slope")

        # ================ 分组历史记录 ================
        
        # 更新历史记录
        self.emb_total_slope_list = torch.cat((self.emb_total_slope_list, emb_total_slope.unsqueeze(0)))
        self.gen_total_slope_list = torch.cat((self.gen_total_slope_list, gen_total_slope.unsqueeze(0)))

        # ================ 分组Divergence Factor计算 ================
        def compute_divergence(slope_history):
            if len(slope_history) < 1:
                return torch.tensor(1.0, device=device)
            
            normalized = -len(slope_history) * slope_history / slope_history.abs().sum()
            divergence = F.softmax(normalized * self.coba_tau, dim=-1)[-1] * len(slope_history)
            return torch.min(divergence, torch.tensor(1.0, device=device))

        emb_divergence = compute_divergence(self.emb_total_slope_list)
        gen_divergence = compute_divergence(self.gen_total_slope_list)

        # ================ 分组权重计算 ================
        def process_group(indices, group_size, divergence):
            # 提取组内参数
            group_slope = task_slope_fitting[indices]
            group_history = self.per_task_slope_list[indices, start_step:]
            
            # RCS计算
            normalized_rcs = group_size * group_slope / group_slope.abs().sum()
            rcs = F.softmax(normalized_rcs, dim=0)
            
            # ACS计算
            reversed_norm = -group_history.shape[1] * group_history / group_history.abs().sum(dim=-1, keepdim=True)
            current_slope = reversed_norm.T.reshape(-1)[-group_size:]
            acs = F.softmax(current_slope, dim=0)
            
            # 组合权重
            weight_logits = divergence * rcs + (1 - divergence) * acs
            if torch.distributed.get_rank() == 0:
                print("RCS:",rcs)
                print("ACS:",acs)
                if group_size == 3:
                    log_embrcs.append(rcs.tolist())
                    log_embacs.append(acs.tolist())
                else:
                    log_genrcs.append(rcs.tolist())
                    log_genacs.append(acs.tolist())

            return F.softmax(weight_logits * group_size, dim=0)

        # 执行分组计算
        emb_weights = process_group(slice(0, emb_num), emb_num, emb_divergence)
        gen_weights = process_group(slice(emb_num, task_num), gen_num, gen_divergence)
        
        # 乘以原始任务数比例
        emb_ratio = torch.tensor(TASK_NUMS[:emb_num],dtype=emb_weights.dtype)
        emb_ratio = emb_ratio / emb_ratio.mean()
        emb_weights = (emb_weights * emb_ratio) / torch.norm(emb_weights * emb_ratio, p=1)

        gen_ratio = torch.tensor(TASK_NUMS[emb_num:],dtype=gen_weights.dtype)
        gen_ratio = gen_ratio / gen_ratio.mean()
        gen_weights = (gen_weights * gen_ratio) / torch.norm(gen_weights * gen_ratio, p=1)


        per_task_weight = torch.cat([emb_weights, gen_weights])

        # ================ 权重修正 ================
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

        # ================ 验证和日志 ================
        if torch.distributed.get_rank() == 0:
            print("Per Task Weight:", per_task_weight)
            print(f"Emb DF: {emb_divergence.item():.2f}, Gen DF: {gen_divergence.item():.2f}")
            log_embdf.append(emb_divergence.item())
            log_gendf.append(gen_divergence.item())

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
        
        
        # if torch.distributed.get_rank() == 0:
        #     print(f'\nValid Task Loss:{self.valid_task_loss_accumulated}')

        # self.valid_task_loss_accumulated /= self.coba_sample_valid_num
        if self.history_task_valid_loss is None and completed_steps >= 1:
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        elif self.history_task_valid_loss is not None:
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )

def save_df(path):
    if torch.distributed.get_rank() == 0:
        df_data = {
            "step": range(len(log_embdf)),
            "emb_df": log_embdf,
            "gen_df": log_gendf
        }
        
        df_path = os.path.join(path, "divergence_factors.csv")
        pd.DataFrame(df_data).to_csv(df_path, index=False)
        print(f"Divergence factors saved to {df_path}")

        # print(task_slope_list)
        print(log_embrcs)
        print(log_embacs)
        print(log_genrcs)
        print(log_genacs)
        # df = pd.DataFrame(task_slope_list.numpy())
        # df_path = os.path.join(path, "task_slope_list.csv")
        # df.to_csv(df_path, index=False, header=False)

        df = pd.DataFrame(log_embrcs)
        df.to_csv(os.path.join(path, "emb_rcs.csv"), index=False, header=False)

        df = pd.DataFrame(log_embacs)
        df.to_csv(os.path.join(path, "emb_acs.csv"), index=False, header=False)

        df = pd.DataFrame(log_genrcs)
        df.to_csv(os.path.join(path, "gen_rcs.csv"), index=False, header=False)

        df = pd.DataFrame(log_genacs)
        df.to_csv(os.path.join(path, "gen_acs.csv"), index=False, header=False)
