# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
import math
import pdb
class loraW(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, A, E, B, scaling, ranknum):
        return torch.cat([b for b in B], 1) @                                 \
                (torch.cat([a for a in A], 0) * torch.cat([e for e in E], 0)) \
                    * scaling / (ranknum+1e-5)

class AdaLoraLayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = ("lora_A", "lora_B", "lora_E", "lora_embedding_A", "lora_embedding_B")
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})
        self.use_increlora: dict[str, bool] = {}
        self.W = nn.ModuleDict({})
        self.score: dict[str, float] = {}
        self.hook_handle = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_increlora=False):
        if r < 0: #fix for r is zero
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        self.use_increlora[adapter_name] = use_increlora
        if use_increlora:
            self.lora_A[adapter_name] = nn.ParameterList(
                [nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, self.in_features))) for _ in range(r)]
            )
            self.lora_E[adapter_name] = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1)) for _ in range(r)]
            )
            self.lora_B[adapter_name] = nn.ParameterList(
                [nn.Parameter(torch.normal(mean=0, std=0.02, size=(self.out_features, 1))) for _ in range(r)]  
            )
            self.W[adapter_name] = loraW()
            self.hook_handle[adapter_name] = self.W[adapter_name].register_full_backward_hook(self.backward_hook)
            self.score[adapter_name] = 0
            #self.gradMatrix_trace[adapter_name] = 0
            
        else:
            # Right singular vectors
            self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
            # Singular values
            self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
            # Left singular vectors
            self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))
        # The current rank
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights and not use_increlora:
            self.reset_lora_parameters(adapter_name)

        if hasattr(self.get_base_layer(), "qweight"):
            # QuantLinear
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)
    def backward_hook(self, module, grad_input, grad_output):
        # print("Output_Grad:", grad_output)
        grad_Matrix = grad_output[0]
        for active_adapter in self.active_adapters:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            scaling = self.scaling[active_adapter]
            ranknum = self.ranknum[active_adapter] + 1e-5
            real_r = int(self.ranknum[active_adapter].data.item())
            try:
                W = (
                    
                    self.W[active_adapter](lora_A[:real_r], lora_E[:real_r], lora_B[:real_r], scaling, ranknum)
                    ).abs()
                # scale_W = torch.mean(W)
                scale_W=1
                self.score[active_adapter] = torch.sum(((W / scale_W) * grad_Matrix).abs().detach()) / math.sqrt(W.numel())
                # self.score = torch.mean((grad_Matrix ** 2).detach())
            except:
                pdb.set_trace()

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class SVDLinear(nn.Module, AdaLoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        use_increlora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_increlora=use_increlora)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if self.use_increlora[adapter]:
            real_r = int(self.ranknum[adapter].data.item())
            return (
                self.W[adapter](self.lora_A[adapter][:real_r], self.lora_E[adapter][:real_r], self.lora_B[adapter][:real_r], self.scaling[adapter], self.ranknum[adapter]).T
            )
        return (
            transpose(self.lora_B[adapter] @ (self.lora_A[adapter] * self.lora_E[adapter]), self.fan_in_fan_out)
            * self.scaling[adapter]
            / (self.ranknum[adapter] + 1e-5)
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-5

                if self.use_increlora[active_adapter]:
                    x = x.to(lora_A[0].dtype)
                    real_r = int(self.ranknum[active_adapter].data.item())
                    try:
                        result += (
                            dropout(x) @ self.W[active_adapter](lora_A[:real_r], lora_E[:real_r], lora_B[:real_r], scaling, ranknum).T
                        )
                    except:
                        pdb.set_trace()
                        print(self.W[active_adapter])
                else:
                    x = x.to(lora_A.dtype)                                         
                    result += (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adalora." + rep


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self.use_increlora = peft_config.use_increlora
        if not self.use_increlora:
            self._set_budget_scheduler(model)
        else:
            self.top_h = self.peft_config.top_h
            self.incre_rank_num = self.peft_config.incre_rank_num
            self._set_budget_scheduler_increlora(model, peft_config.init_r)
            

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def _set_budget_scheduler_increlora(self, model, init_r):
        import math
        self.init_bgt = 0
        self.name_set = set()
        for name,module in model.named_modules():
            if hasattr(module,"lora_E"):
                self.init_bgt += init_r
                # module.lora_A[self.adapter_name].data[:init_r,:] = torch.randn_like(module.lora_A[self.adapter_name].data[:init_r,:]) * 0.02
                # module.lora_A[self.adapter_name].data[init_r:,:] = torch.zeros_like(module.lora_A[self.adapter_name].data[init_r:,:])
                self.name_set.add(name)
                # module.lora_B[self.adapter_name].data[:,init_r:] = torch.zeros_like(module.lora_B[self.adapter_name].data[:,init_r:])
                # module.lora_B[self.adapter_name].data[:,:init_r] = torch.randn_like(module.lora_B[self.adapter_name].data[:,:init_r]) * 0.02
                module.ranknum[self.adapter_name].data.fill_(float(init_r))

        self.name_set = sorted(self.name_set)
        self.target_bgt = self.peft_config.target_r * len(self.name_set)
        rank_per_round = self.top_h * self.incre_rank_num
        total_round = math.ceil((self.target_bgt - self.init_bgt) / rank_per_round)
        total_incre_step = self.peft_config.deltaT * total_round
        print("Total incremental step: total_incre_step: {}, of total steps: {:.0%}"
              .format(total_incre_step, total_incre_step / (self.peft_config.tfinal - self.peft_config.tinit)))
    

    # The budget schedule for AdaLora
    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        if not self.use_increlora:             
            # Update the sensitivity and uncertainty for every weight
            for n, p in model.named_parameters():
                if "lora_" in n and self.adapter_name in n:
                    if n not in self.ipt:
                        self.ipt[n] = torch.zeros_like(p)
                        self.exp_avg_ipt[n] = torch.zeros_like(p)
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                    with torch.no_grad():
                        self.ipt[n] = (p * p.grad).abs().detach()
                        # Sensitivity smoothing
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        # Uncertainty quantification
                        self.exp_avg_unc[n] = (
                            self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        )
        else:
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    if n not in self.ipt:
                        self.ipt[n] = 0
                        self.exp_avg_ipt[n] = 0
                        self.exp_avg_unc[n] = 0
                    
                    # self.tb_writter.add_scalar("GradMatrix_Rank/%s"%(n[:-7],), layer.gradMatrix_rank, global_step)
                    try:
                        self.ipt[n] = layer.score["default"]
                    
                        # Update sensitivity 
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                            (1-self.beta1)*self.ipt[n]
                        # Update uncertainty 
                        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                            (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()
                    except:
                        pdb.set_trace()
                        print(layer)

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt
    def calculate_score(self, n, layer, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = 0.
            for n,p in layer.named_parameters():
                ipt_score += p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        if not self.use_increlora:
            # Get the importance score for A, E, B
            for n, p in model.named_parameters():
                if f"lora_A.{self.adapter_name}" in n:
                    entry_ipt = self._element_score(n)
                    comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                    name_m = n.replace("lora_A", "%s")
                    if name_m not in vector_ipt:
                        vector_ipt[name_m] = [comb_ipt]
                    else:
                        vector_ipt[name_m].append(comb_ipt)
                if f"lora_B.{self.adapter_name}" in n:
                    entry_ipt = self._element_score(n)
                    comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                    name_m = n.replace("lora_B", "%s")
                    if name_m not in vector_ipt:
                        vector_ipt[name_m] = [comb_ipt]
                    else:
                        vector_ipt[name_m].append(comb_ipt)
                if f"lora_E.{self.adapter_name}" in n:
                    entry_ipt = self._element_score(n)
                    name_m = n.replace("lora_E", "%s")
                    value_ipt[name_m] = entry_ipt
            all_score = []
        
             # Calculate the score for each triplet
            for name_m in vector_ipt:
                ipt_E = value_ipt[name_m] #shape=8,1
                ipt_AB = torch.cat(vector_ipt[name_m], dim=1) #shape=8,2
                sum_ipt = self._combine_ipt(ipt_E, ipt_AB) #shape=8
                name_E = name_m % "lora_E"
                triplet_ipt[name_E] = sum_ipt.view(-1, 1) #shape=8,1
                all_score.append(sum_ipt.view(-1))
            # Get the threshold by ranking ipt
            mask_threshold = torch.kthvalue(
                torch.cat(all_score),
                k=self.init_bgt - budget,
            )[0].item()
        else:
            current_bgt = 0
            triplet_ipt = {}
            all_score = []
            # Calculate the importance score for each sub matrix 
            for n, layer in model.named_modules():
                if isinstance(layer, SVDLinear):
                    ipt_score = self.calculate_score(n, layer, metric="ipt")
                    r = int(layer.ranknum[self.adapter_name].data.item())
                    current_bgt += r             
                    triplet_ipt[n] = ipt_score
                    all_score.append(ipt_score)
            # for name_m in vector_ipt:
            #     ipt_E = value_ipt[name_m] #shape=8,1
            #     ipt_AB = torch.cat(vector_ipt[name_m], dim=1) #shape=8,2
            #     sum_ipt = self._combine_ipt(ipt_E, ipt_AB) #shape=8
            #     name_E = name_m % "lora_E"
            #     non_zero_elements = sum_ipt[sum_ipt != 0]
            #     current_bgt += non_zero_elements.size(0)
            #     triplet_ipt[name_E] = (torch.mean(non_zero_elements))
            #     all_score.append(torch.mean(non_zero_elements).unsqueeze(0))
            k = min(self.top_h, int((self.target_bgt - current_bgt)/self.incre_rank_num))
            if k<=0:
                increase_threshold = torch.tensor(100000)
            else:
                increase_threshold = torch.topk(torch.cat(all_score), k)[0][-1].item() 


        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            if not self.use_increlora:
                for n, p in model.named_parameters():
                    if f"lora_E.{self.adapter_name}" in n:
                        p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                        rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
            else:
                for name,module in model.named_modules():
                    if isinstance(layer, SVDLinear):
                        if triplet_ipt[name] >= increase_threshold:
                            add_r = self.incre_rank_num
                            old_r = int(module.ranknum[self.adapter_name].data.item())
                            # module.lora_A[self.adapter_name].data[old_r:old_r+add_r,:] = torch.randn_like( module.lora_A[self.adapter_name].data[old_r:old_r+add_r,:]) * 0.02
                            # module.lora_B[self.adapter_name].data[:,old_r:old_r+add_r] = torch.randn_like( module.lora_B[self.adapter_name].data[:,old_r:old_r+add_r]) * 0.02
                            module.ranknum[self.adapter_name].data += add_r
                            total_r = old_r + add_r
                            module.scaling[self.adapter_name] = 2*float(total_r)
                            pattern = [True] * total_r + [False] * (module.lora_A[self.adapter_name].size(0) - total_r)
                            rank_pattern[name+f".lora_E.{self.adapter_name}"] = pattern
                        else:
                            old_r = int(module.ranknum[self.adapter_name].data.item())
                            pattern = [True] * old_r + [False] * (module.lora_A[self.adapter_name].size(0) - old_r)
                            rank_pattern[name+f".lora_E.{self.adapter_name}"] = pattern
                        
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)
