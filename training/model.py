from dataclasses import dataclass
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
from peft import PeftModel
import torch.nn.functional as F
from utils.common_utils import ID2TASK, TASK2ID

from gritlm import GritLM

logger = logging.getLogger(__name__)


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps, p_reps, loss_weight, emb_task):

        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        return self.cross_entropy(scores, target)
    
    def _dist_gather_task(self, task_ids):
        """任务ID专用收集方法"""
        task_ids = task_ids.contiguous()
        gather_tasks = [torch.empty_like(task_ids) for _ in range(self.world_size)]
        dist.all_gather(gather_tasks, task_ids)
        gather_tasks[self.rank] = task_ids
        return torch.cat(gather_tasks)

    def _dist_gather_weight(self, weight):
        """专为1D权重设计的收集方法"""
        weight = weight.contiguous()
        weight_list = [torch.empty_like(weight) for _ in range(self.world_size)]
        dist.all_gather(weight_list, weight)
        weight_list[self.rank] = weight  # 避免重复
        return torch.cat(weight_list)  # [world_size * bs]

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2: return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        if loss_gen_type == "token": # b.1)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed": # c)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        
    def __call__(self, labels, logits, loss_weight, gen_task):

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # Normalize by number of non-ignored tokens
        if self.loss_gen_type == "token":
            return (self.cross_entropy(shift_logits, shift_labels) / labels.size(0)) * self.loss_gen_factor
        elif self.loss_gen_type == "mixed":
            return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor


class GritLMTrainModel(GritLM):
    TRANSFORMER_CLS = AutoModel
    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device)
        self.gen_add_kwargs = {"return_dict": True}
        if "mixtral" in kwargs["model_name_or_path"].lower():
            logger.info("Using token loss with routing loss for mixtral")
            self.gen_loss_fn = None
            self.gen_add_kwargs["loss_gen_factor"] = loss_gen_factor
            self.gen_add_kwargs["output_router_logits"] = True
        else:
            self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )
            
        # Initialization For Embedding Table
        def weight_init(m):
            if isinstance(m, torch.nn.Embedding):
                torch.nn.init.xavier_normal_(m.weight)
            
        self.config = self.model.config # Required for accelerate DeepSpeed integration
        self.task_embedding = self.task_embedding.apply(weight_init)

    def encode(self, features, task_types):
        if features is None: return None
        # Clone to avoid modifying the original tensor
        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        instruction_lens = features['instruction_lens'] if 'instruction_lens' in features else None
        if task_types is None:
            task_index = None
        else:
            task_index = torch.zeros(task_types.shape[0],1).to(self.device)
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask, 'task_types': task_types, 'task_index':task_index}

        if self.attn[:2] == 'cb':
            kwargs['instruction_lens'] = instruction_lens
        elif self.attn[:2] == 'bb':
            kwargs['is_causal'] = False
            
        if isinstance(self.model, PeftModel):
            out = (getattr(self.model.base_model.model, self.embedding_attr) if self.embedding_attr else self.model.base_model.model)(**kwargs)[0]
        else:
            out = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**kwargs)[0]
        # out = self.model(**kwargs)[0]

        if self.projection is not None:
            out = self.projection(out)
        
        # Mask out the instruction tokens for pooling
        if instruction_lens is not None:
            # Make a new copy of attention mask to prevent in-place problems
            attention_mask = features['attention_mask'].clone()
            # Mask out the instruction tokens for pooling
            for i, l in enumerate(instruction_lens):
                attention_mask[i, :l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, f"All 0: {attention_mask[i]}, l: {l}"

        reps = self.pooling(out, attention_mask)
        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized: 
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        generative: Dict[str, torch.Tensor] = None,
        q_reps: Optional[torch.Tensor] = None,
        p_reps: Optional[torch.Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
        emb_task: Optional[torch.Tensor] = None,
        gen_task: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query: [b, n]
            passage: [b*s, m] where s is group size (usually 2)
            generative: [b, m]
        """
        # Do generative first, as emb contains an all-reduce (verified to be faster)
        if generative is not None:
            if gen_task is None:
                gen_task = generative.pop('gen_task')
            if gen_task is not None:
                task_types = self.task_embedding(gen_task.to(self.device))
                task_index = torch.ones(task_types.shape[0],1).to(task_types.device)
            else:
                task_index = None
                
            if loss_weight is not None:
                self.loss_weight = loss_weight.to(self.device)
            
            if self.gen_loss_fn is not None:
                # This pops the labels first, then the rest is passed into model                
                loss_gen = self.gen_loss_fn(
                    generative.pop('labels'), 
                    self.model(**generative, task_types=task_types, task_index=task_index, **self.gen_add_kwargs).logits,
                    loss_weight,
                    gen_task,
                )
            else:
                loss_gen = self.model(**generative, task_types=task_types, task_index=task_index, **self.gen_add_kwargs).loss
        else:
            loss_gen = None
               
        if query is not None and emb_task is None:
            emb_task = query.pop("emb_task")
            if loss_weight is None:
                loss_weight = self.loss_weight
        
        if emb_task is not None:
            task_emb = self.task_embedding(emb_task.to(self.device))

        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query, task_types=task_emb)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query, task_types=task_emb)

        if (p_reps is None) and (passage is not None):
            if p_grad:
                p_reps = self.encode(passage, task_types=task_emb)
            else:
                with torch.no_grad():
                    p_reps = self.encode(passage, task_types=task_emb)
            
        loss_emb = self.emb_loss_fn(
            q_reps, p_reps, loss_weight, emb_task
        ) if (q_reps is not None and p_reps is not None) else None        

        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])

        # Also return q_reps in case of GradCache
        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)   #Modified
