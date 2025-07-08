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
from __future__ import annotations
import torch.nn.functional as F
import json
import math
import time
import os
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from collections import defaultdict
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
import copy

class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        # dict for log
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.old_state_dict = None
        self.old_optim_state = None

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)


        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "use_mora": lora_config.use_mora,
            "use_beamlora": lora_config.use_beamlora,
            "use_pissa": lora_config.use_pissa,
            "use_erank": lora_config.use_erank,
            "plora_m": lora_config.plora_m,
            "mora_type": lora_config.mora_type,
            "record_activations": lora_config.record_activations,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }


        use_mora = lora_config.use_mora

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                use_mora=use_mora,
                use_pissa=lora_config.use_pissa,
                use_beamlora=lora_config.use_beamlora,
                use_erank=lora_config.use_erank,
                plora_m=lora_config.plora_m,
                mora_type=lora_config.mora_type,
                record_activations=lora_config.record_activations,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend([dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_megatron, dispatch_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LORA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
        density=None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] == set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
    
    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        # import pdb
        # pdb.set_trace()

        # l1_lambda = 1e-2
        # if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor) and self.peft_config['default'].use_beamlora:
        #     l1_loss = 0
        #     with torch.no_grad():
        #         for n, p in self.model.named_parameters():
        #             if "lora_gate" in n:
        #                 #l1_loss += torch.sum(torch.abs(p+torch.ones_like(p)/10))
        #                 l1_loss += torch.sum(torch.abs(p))
        #     #print(l1_loss)

        #     outputs.loss += l1_lambda * l1_loss



        # Calculate the orthogonal regularization
        #orth_reg_weight = 0.1

        # if orth_reg_weight <= 0:
        #     raise ValueError("orth_reg_weight should be greater than 0. ")

        # regu_loss = 0
        # num_param = 0
        # with torch.no_grad():
        #     for n, p in self.model.named_parameters():
        #         if ("lora_A" in n or "lora_B" in n) and self.peft_config['default'].use_beamlora:
        #             para_cov = p @ p.T if "lora_A" in n else p.T @ p
        #             I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))  # noqa: E741
        #             I.requires_grad = False
        #             num_param += 1
        #             regu_loss += torch.norm(para_cov - I, p="fro")
        #     if num_param > 0:
        #         regu_loss = regu_loss / num_param
        #     else:
        #         regu_loss = 0
        #print(regu_loss)
        #outputs.loss += orth_reg_weight * regu_loss
        return outputs
    def update_ipt(self, beta1=0.85, beta2=0.85):
        # Update the sensitivity and uncertainty for every weight
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = beta1 * self.exp_avg_ipt[n] + (1 - beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        beta2 * self.exp_avg_unc[n] + (1 - beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )
    def gate_imp_softmax(self, p, temperature = 0.05):    # 0.05 for lr>=1e-4, 0.03 for lr<1e-4
        return F.softmax(p/temperature)*p.shape[1]
    
    def first_order_importance(self):
        def _element_score(n):
            return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

        def _combine_ipt(ipt_AB):
            ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
            return ipt_AB.view(-1)
        result_ipt = {}
        vector_ipt = {}
        # Get the importance score for A, E, B
        for n, p in self.model.named_parameters():
            if f"lora_A" in n:
                entry_ipt = _element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace(".lora_A.default.weight", "")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B" in n:
                entry_ipt = _element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace(".lora_B.default.weight", "")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)

        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1) #shape=8,2
            sum_ipt = _combine_ipt(ipt_AB) #shape=8
            result_ipt[name_m] = sum_ipt.unsqueeze(0) #shape=1,8
        return result_ipt
    def modify_above_half_mask(self, masks, values):
        """
        Update masks based on values, making sure no more than half the elements are True.

        Parameters:
        masks (dict): Dictionary where the key is the module name and the value is a torch tensor of bools.
        values (dict): Dictionary where the key is the module name and the value is a torch tensor corresponding to masks.

        Returns:
        int: The total number of True values in the updated masks dictionary.
        """
        for module_name in masks:
            mask = masks[module_name][0]
            value = values[module_name][0]
            
            true_count = mask.sum().item()  # Count how many are True
            n = mask.numel()
            max_true = n // 2  # Maximum allowed True values
            
            if true_count > max_true:
                # Calculate how many Trues need to be set to False
                excess_true_count = true_count - max_true
                
                # Get indices of True elements
                true_indices = torch.where(mask)[0]
                
                # Sort these indices based on the corresponding values in ascending order
                sorted_indices = true_indices[value[true_indices].argsort(descending=True)]
                
                # Set the first `excess_true_count` True elements to False
                mask[sorted_indices[:excess_true_count]] = False
                masks[module_name][0] = mask
                
        # Calculate the updated total number of True values across all modules
        total_true_count = sum(mask.sum().item() for mask in masks.values())
        print("After modify, total number of True values in masks:", total_true_count)
        return masks


    def cache_state(self,optimizer_state):
        del self.old_state_dict
        del self.old_optim_state
        old_state_dict = {}
        old_optim_state = {}
        for n, p in self.model.named_parameters():
            if f"lora" in n:
                old_state_dict[n] = p.data.clone()
                old_optim_state[n] = copy.deepcopy(optimizer_state.state[p])
        self.old_state_dict = old_state_dict
        self.old_optim_state = old_optim_state
    def analyze_rank_importance(self, threshold, min_k=True, global_step=None, output_path=None, 
        evaluation_method="gate_value", optimizer_states=None, has_gate=True, activation_temperature=1,
        reverse_imp=False,use_top_p=True,cumsum_threshold=0.95,random_mask=False,constant_K=False) -> None:
        print("min_k",str(min_k))
        print("evaluation_method",str(evaluation_method))
        if min_k:
            threshold = threshold
        else:
            threshold = 1 - threshold
        value_ipt = {}
        masks={}
        all_score = []
        num_lora_ranks = 0
        def count_total_trues(masks):
            total_trues = 0
            for layer_name, mask in masks.items():
                trues_in_layer = torch.sum(mask).item()
                total_trues += trues_in_layer
            print("Total number of True values in masks:", total_trues)
        def get_gate_value_imp():
            for n, p in self.model.named_parameters():
                if f"lora_gate." in n:
                    entry_ipt = self.gate_imp_softmax(p).detach()
                    value_ipt[n] = entry_ipt
                    all_score.append(entry_ipt.view(-1))
                    #num_lora_ranks += entry_ipt.shape[1]

        def get_gratient_imp_wtih_gate(active_adapter):
            for n, p in self.model.named_parameters():
                if f"lora_gate." in n:
                    _, target, _ = _get_submodules(self.model, n.replace(".lora_gate."+active_adapter,""))
                    weight_A_state = optimizer_states.state[target.lora_A[active_adapter].weight]
                    weight_B_state = optimizer_states.state[target.lora_B[active_adapter].weight]
                    entry_ipt = (torch.mean(weight_A_state["exp_avg_sq"],dim=1)+torch.mean(weight_B_state["exp_avg_sq"],dim=0)).unsqueeze(0)
                    value_ipt[n] = entry_ipt
                    all_score.append(entry_ipt.view(-1))
                    #num_lora_ranks += target.lora_A[active_adapter].weight.shape[0]
        def get_act_imp_wo_gate(active_adapter):
            for name, module in self.model.named_modules():
                if hasattr(module, "recorded_activations"):
                    entry_ipt = F.softmax(module.recorded_activations.unsqueeze(0))
                    value_ipt[name] = entry_ipt
                    all_score.append(entry_ipt.view(-1))

        def get_act_imp_cumsum(active_adapter,cumsum_threshold=0.9, only_do_on_ffn=False):
            print("activation_temperature:",str(activation_temperature))
            print("cumsum_threshold:",str(cumsum_threshold))
            print("reverse_imp:",str(reverse_imp))
            print("only_do_on_ffn",str(only_do_on_ffn))
            for name, module in self.model.named_modules():
                if hasattr(module, "recorded_activations"):
                    importance = F.softmax(module.recorded_activations/activation_temperature)
                    if reverse_imp:
                        mean_values = importance.mean()
                        symmetric_p = 2 * mean_values - importance
                        importance = symmetric_p

                    # Step 1: Sort the importance in descending order and get the sorted indices
                    sorted_indices = torch.argsort(importance, descending=True)
                    sorted_importance = importance[sorted_indices]
                    # Step 2: Compute cumulative sum of the sorted importance
                    cumulative_importance = torch.cumsum(sorted_importance, dim=0)
                    # Step 3: Find the position where cumulative importance exceeds the threshold
                    mask_position = (cumulative_importance > cumsum_threshold).nonzero(as_tuple=True)[0]

                    # Step 4: Create a mask with the same size as importance
                    mask = torch.zeros_like(importance, dtype=torch.bool)

                    # Step 5: Mask the positions corresponding to the indices beyond the threshold
                    if mask_position.numel() > 0:
                        mask_position = mask_position[0]
                        important_indices = sorted_indices[mask_position:]
                        mask[important_indices] = True
                    if only_do_on_ffn:
                        if ("mlp.up_proj" in name) or ("mlp.gate_proj" in name):
                            pass
                        else:
                            mask = torch.zeros_like(importance, dtype=torch.bool)
                    masks[name] = mask.unsqueeze(0)
                    value_ipt[name] = importance.unsqueeze(0)
        def get_act_imp_range(active_adapter,range_threshold=0.9):
            print("activation_temperature:",str(activation_temperature))
            print("cumsum_threshold:",str(cumsum_threshold))
            
            for name, module in self.model.named_modules():
                if hasattr(module, "recorded_activations"):
                    importance = F.softmax(module.recorded_activations/activation_temperature)
                    # Step 1: Sort the importance in descending order and get the sorted indices
                    sorted_indices = torch.argsort(importance, descending=True)
                    sorted_importance = importance[sorted_indices]
                    # Step 2: Compute cumulative sum of the sorted importance
                    cumulative_importance = torch.cumsum(sorted_importance, dim=0)
                    # Step 3: Find the position where cumulative importance exceeds the threshold
                    mask_position = (cumulative_importance > cumsum_threshold).nonzero(as_tuple=True)[0]

                    # Step 4: Create a mask with the same size as importance
                    mask = torch.zeros_like(importance, dtype=torch.bool)
                    
                    # Step 5: Mask the positions corresponding to the indices beyond the threshold
                    if mask_position.numel() > 0:
                        mask_position = mask_position[0]
                        important_indices = sorted_indices[mask_position:]
                        mask[important_indices] = True
                    

                    masks[name] = mask.unsqueeze(0)
                    value_ipt[name] = importance.unsqueeze(0)
        def get_first_order_imp(active_adapter, use_softmax=True):
            value_ipt = self.first_order_importance()
            for key in value_ipt.keys():
                if use_softmax:
                    value_ipt[key] = F.softmax(value_ipt[key]/torch.mean(value_ipt[key]))
                all_score.append(value_ipt[key].view(-1))
            return value_ipt
            
        with torch.no_grad():
            for active_adapter in self.active_adapters:
                if evaluation_method=="gate_value": # Caculate IMP by gate value
                    get_gate_value_imp()
                    num_lora_ranks = torch.cat(all_score).shape[0]
                    print(num_lora_ranks)
                    # Get the threshold by ranking ipt
                    if use_top_p:
                        importance = torch.cat(all_score)
                        # Step 1: Sort the importance in descending order and get the sorted indices
                        sorted_indices = torch.argsort(importance, descending=True)
                        sorted_importance = importance[sorted_indices]
                        # Step 2: Compute cumulative sum of the sorted importance
                        cumulative_importance = torch.cumsum(sorted_importance, dim=0)
                        # Step 3: Find the position where cumulative importance exceeds the threshold
                        mask_position = (cumulative_importance > cumsum_threshold * len(value_ipt) * self.peft_config[active_adapter].r).nonzero(as_tuple=True)[0]

                        # Step 4: Create a mask with the same size as importance
                        concat_mask = torch.zeros_like(importance, dtype=torch.bool)

                        # Step 5: Mask the positions corresponding to the indices beyond the threshold
                        if mask_position.numel() > 0:
                            mask_position = mask_position[0]
                            important_indices = sorted_indices[mask_position:]
                            concat_mask[important_indices] = True
                        print(torch.sum(concat_mask == True))
                        start_dim = 0
                        for key in value_ipt.keys():
                            masks[key] = concat_mask[start_dim : start_dim + value_ipt[key].shape[1]].unsqueeze(0)
                            start_dim += value_ipt[key].shape[1]
                    else:     
                        print(int(num_lora_ranks*threshold))
                        mask_threshold = torch.kthvalue(
                            torch.cat(all_score),
                            k=int(num_lora_ranks*threshold),
                        )[0].item()
                        masks = {n: (value <= mask_threshold if min_k else value >= mask_threshold) 
                                for n, value in value_ipt.items()}
                elif evaluation_method=="gradient": # Caculate IMP by gradient
                    get_gratient_imp_wtih_gate(active_adapter)
                    # Get the threshold by ranking ipt
                    num_lora_ranks = torch.cat(all_score).shape[0]
                    print(num_lora_ranks)
                    print(int(num_lora_ranks*threshold))
                    mask_threshold = torch.kthvalue(
                        torch.cat(all_score),
                        k=int(num_lora_ranks*threshold),
                    )[0].item()
                    masks = {n: (value <= mask_threshold if min_k else value >= mask_threshold) 
                            for n, value in value_ipt.items()}
                elif evaluation_method=="activation": # Caculate IMP by activation
                    #get_act_imp_wo_gate(active_adapter)
                    get_act_imp_cumsum(active_adapter)
                    count_total_trues(masks)
                elif evaluation_method=="first_order":
                    value_ipt = get_first_order_imp(active_adapter)
                    # Get the threshold by ranking ipt
                    num_lora_ranks = torch.cat(all_score).shape[0]
                    print(num_lora_ranks)
                    print(int(num_lora_ranks*threshold))
                    mask_threshold = torch.kthvalue(
                        torch.cat(all_score),
                        k=int(num_lora_ranks*threshold),
                    )[0].item()
                    
                    masks = {n: (value <= mask_threshold if min_k else value >= mask_threshold) 
                            for n, value in value_ipt.items()}
                    masks = self.modify_above_half_mask(masks=masks, values=value_ipt)
                rank_pattern_save = {n: (~mask).view(-1).tolist() for n,mask in masks.items()}
                value_ipt_save = {n: value.tolist() for n, value in value_ipt.items()}

        directory_path = output_path+'/rank_logs'
        try:
            os.mkdir(directory_path)
            print(f"Dictionary created successfully at path '{directory_path}'")
        except FileExistsError:
            print(f"Dictionary already exists at path '{directory_path}'")
        with open(directory_path+'/imp_value_'+str(global_step)+'.json', 'w') as f:
            json.dump(value_ipt_save, f)
        with open(directory_path+"/rank_pattern_"+str(global_step)+'.json', 'w') as f:
            json.dump(rank_pattern_save, f)
        return masks, value_ipt

    def reset_rank(self, merge_to_base_model=False, value_ipt=None, optimizer_states=None, masks=None, inherit_param=False, reverse_gate=False, reset_param=False, has_gate=True, only_pruning=False, random_mask=False) -> None:
        reset_rank_sum = 0
        print("merge_to_base_model",str(merge_to_base_model))
        print("reset_param",str(reset_param))
        def inherit_param_ckpt_wo_gate(value, mask, target, active_adapter, name):
            if self.old_state_dict is not None:
                original_weight_A = self.old_state_dict[name+".lora_A."+active_adapter+".weight"].clone()
                original_weight_B = self.old_state_dict[name+".lora_B."+active_adapter+".weight"].clone()
            else:
                original_weight_A = target.lora_A[active_adapter].weight.data.clone()
                original_weight_B = target.lora_B[active_adapter].weight.data.clone()
            top_n_values, top_n_indices = torch.topk(value, int(torch.sum(mask == True)), dim=1)
            
            num_indices = top_n_indices.size(1)
            
            shuffled_indices = torch.randperm(num_indices)
            
            
            shuffled_top_n_indices = top_n_indices[:,shuffled_indices]
            target.lora_A[active_adapter].weight.data[mask[0],:] = original_weight_A[shuffled_top_n_indices[0],:]
            target.lora_B[active_adapter].weight.data[:,mask[0]] = original_weight_B[:,shuffled_top_n_indices[0]]

            target.lora_A[active_adapter].weight.data[mask[0],:] /= 2
            target.lora_A[active_adapter].weight.data[shuffled_top_n_indices[0],:] /= 2
        
        def inherit_param_ckpt(value, mask, target, active_adapter, name):
            lora_A_name = name.replace(".lora_gate."+active_adapter,".lora_A."+active_adapter+".weight")
            lora_B_name = name.replace(".lora_gate."+active_adapter,".lora_B."+active_adapter+".weight")
            original_weight_A = self.old_state_dict[lora_A_name].clone()
            original_weight_B = self.old_state_dict[lora_B_name].clone()
            top_n_values, top_n_indices = torch.topk(value, int(torch.sum(mask == True)), dim=1)
            
            num_indices = top_n_indices.size(1)
            
            shuffled_indices = torch.randperm(num_indices)
            
            shuffled_top_n_indices = top_n_indices[:,shuffled_indices]

            if optimizer_states:
                #print("reset optimizer.")
                weight_A_state = optimizer_states.state[target.lora_A[active_adapter].weight]
                weight_B_state = optimizer_states.state[target.lora_B[active_adapter].weight]
                weight_G_state = optimizer_states.state[target.lora_gate[active_adapter]]
                # #A
                # weight_A_state["exp_avg"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg"][shuffled_top_n_indices[0],:])
                # weight_A_state["exp_avg_sq"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg_sq"][shuffled_top_n_indices[0],:])
                # #B
                # weight_B_state["exp_avg"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg"][:,shuffled_top_n_indices[0]])
                # weight_B_state["exp_avg_sq"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg_sq"][:,shuffled_top_n_indices[0]])

                #A
                weight_A_state["exp_avg"][mask[0],:].copy_(self.old_optim_state[lora_A_name]["exp_avg"][shuffled_top_n_indices[0],:])
                weight_A_state["exp_avg_sq"][mask[0],:].copy_(self.old_optim_state[lora_A_name]["exp_avg_sq"][shuffled_top_n_indices[0],:])
                #B
                weight_B_state["exp_avg"][:,mask[0]].copy_(self.old_optim_state[lora_B_name]["exp_avg"][:,shuffled_top_n_indices[0]])
                weight_B_state["exp_avg_sq"][:,mask[0]].copy_(self.old_optim_state[lora_B_name]["exp_avg_sq"][:,shuffled_top_n_indices[0]])

                #Gate
                weight_G_state["exp_avg"].copy_(torch.zeros_like(weight_G_state["exp_avg"]))
                weight_G_state["exp_avg_sq"].copy_(torch.zeros_like(weight_G_state["exp_avg_sq"]))
            
            
            #original_weight_A[shuffled_top_n_indices[0],:] /= 2
            #target.lora_A[active_adapter].weight.data[shuffled_top_n_indices[0],:] = original_weight_A[shuffled_top_n_indices[0],:]
            target.lora_A[active_adapter].weight.data[mask[0],:] = original_weight_A[shuffled_top_n_indices[0],:]
            target.lora_B[active_adapter].weight.data[:,mask[0]] = original_weight_B[:,shuffled_top_n_indices[0]]


            # gate avg
            target.lora_gate[active_adapter].data[:,mask[0]]=(target.lora_gate[active_adapter].data[:,mask[0]]+target.lora_gate[active_adapter].data[:,shuffled_top_n_indices[0]])/2
            target.lora_gate[active_adapter].data[:,shuffled_top_n_indices[0]] = target.lora_gate[active_adapter].data[:,mask[0]]

        def inherit_param_zeros(value,mask,target,active_adapter):
            original_weight_A = target.lora_A[active_adapter].weight.data.clone()
            original_weight_B = target.lora_B[active_adapter].weight.data.clone()
            top_n_values, top_n_indices = torch.topk(value, int(torch.sum(mask == True)), dim=1)
            
            num_indices = top_n_indices.size(1)
            
            shuffled_indices = torch.randperm(num_indices)
            
            
            shuffled_top_n_indices = top_n_indices[:,shuffled_indices]
            target.lora_A[active_adapter].weight.data[mask[0],:] = torch.zeros_like(original_weight_A[shuffled_top_n_indices[0],:])
            target.lora_B[active_adapter].weight.data[:,mask[0]] = torch.zeros_like(original_weight_B[:,shuffled_top_n_indices[0]])

        def inherit_param_merge(value, mask, target, active_adapter):
            original_weight_A = target.lora_A[active_adapter].weight.data.clone()
            original_weight_B = target.lora_B[active_adapter].weight.data.clone()
            mask_1 = mask[0]
            
            
            false_indices = torch.where(mask_1 == False)[0]

            
            true_indices = torch.where(mask_1 == True)[0]
            for true_idx in true_indices:
                
                random_indices = torch.randperm(len(false_indices))[:2]
                idx1, idx2 = false_indices[random_indices]

                
                avg_weight_A = (original_weight_A[idx1, :] + original_weight_A[idx2, :]) / 2
                avg_weight_B = (original_weight_B[:, idx1] + original_weight_B[:, idx2]) / 2

                
                target.lora_A[active_adapter].weight.data[true_idx, :] = avg_weight_A
                target.lora_B[active_adapter].weight.data[:, true_idx] = avg_weight_B

        def inherit_param(value,mask,target,active_adapter):
            original_weight_A = target.lora_A[active_adapter].weight.data.clone()
            original_weight_B = target.lora_B[active_adapter].weight.data.clone()
            top_n_values, top_n_indices = torch.topk(value, int(torch.sum(mask == True)), dim=1)
            
            num_indices = top_n_indices.size(1)
            
            shuffled_indices = torch.randperm(num_indices)
            
            
            shuffled_top_n_indices = top_n_indices[:,shuffled_indices]
            target.lora_A[active_adapter].weight.data[mask[0],:] = original_weight_A[shuffled_top_n_indices[0],:]
            target.lora_B[active_adapter].weight.data[:,mask[0]] = original_weight_B[:,shuffled_top_n_indices[0]]
            if hasattr(target,"lora_gate"):
                target.lora_gate[active_adapter].data[:,shuffled_top_n_indices[0]] /= 2
                target.lora_gate[active_adapter].data[:,mask[0]] = target.lora_gate[active_adapter].data[:,shuffled_top_n_indices[0]]
            else:
                target.lora_A[active_adapter].weight.data[mask[0],:] /= 2
                target.lora_A[active_adapter].weight.data[shuffled_top_n_indices[0],:] /= 2

    
        def reset_rank_with_gate(active_adapter, reset_all_gate_optimizer=True, use_old_param_state=True):
            print("reset_all_gate_optimizer",str(reset_all_gate_optimizer))
            print("use_old_param_state",str(use_old_param_state))
            
            for n, p in self.model.named_parameters():
                if f"lora_gate." in n:
                    
                    mask = masks[n]
                    original_weight_gate = self.gate_imp_softmax(p).detach()
                    _, target, _ = _get_submodules(self.model, n.replace(".lora_gate."+active_adapter,""))
                    original_weight_A = target.lora_A[active_adapter].weight.data.clone()
                    original_weight_B = target.lora_B[active_adapter].weight.data.clone()
                    if inherit_param:
                        if use_old_param_state:
                            inherit_param_ckpt(original_weight_gate,mask,target,active_adapter,n)
                        else:
                            inherit_param(original_weight_gate,mask,target,active_adapter)
                    elif reverse_gate: 
                        mean_values = p.mean()
                        symmetric_p = 2 * mean_values - p
                        p = symmetric_p
                    else:
                        p.masked_fill_(mask, -100)
                        non_zero_mean = p[p != -100].mean().item()
                        if math.isnan(non_zero_mean):
                            non_zero_mean = 0.0
                        p[mask] = non_zero_mean    
                    merge_percentage = "half"
                    if reset_param:
                        target.lora_A[active_adapter].weight.data[mask[0],:] = nn.init.kaiming_uniform_(torch.zeros_like(target.lora_A[active_adapter].weight.data[mask[0],:]), a=math.sqrt(5))
                        target.lora_B[active_adapter].weight.data[:,mask[0]] = torch.zeros_like(target.lora_B[active_adapter].weight.data[:,mask[0]])
                        merge_percentage = "full"
                    # if optimizer_states:
                    #     print("reset optimizer.")
                    #     # import pdb
                    #     # pdb.set_trace()
                    #     weight_A_state = optimizer_states.state[target.lora_A[active_adapter].weight]
                    #     weight_B_state = optimizer_states.state[target.lora_B[active_adapter].weight]
                    #     weight_G_state = optimizer_states.state[p]
                    #     weight_A_state["exp_avg"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg"][mask[0],:])
                    #     weight_A_state["exp_avg_sq"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg_sq"][mask[0],:])
                    #     weight_B_state["exp_avg"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg"][:,mask[0]])
                    #     weight_B_state["exp_avg_sq"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg_sq"][:,mask[0]])
                    #     if reset_all_gate_optimizer:
                    #         weight_G_state["exp_avg"] = torch.zeros_like(weight_G_state["exp_avg"])
                    #         weight_G_state["exp_avg_sq"] = torch.zeros_like(weight_G_state["exp_avg_sq"])
                    #     else:
                    #         weight_G_state["exp_avg"][:,mask[0]] = torch.zeros_like(weight_G_state["exp_avg"][:,mask[0]])
                    #         weight_G_state["exp_avg_sq"][:,mask[0]] = torch.zeros_like(weight_G_state["exp_avg_sq"][:,mask[0]])
                    if merge_to_base_model and mask.sum() > 0:
                        #print(mask.sum())
                        reset_rank_sum += mask.sum()
                        merge_lora_A_vec = original_weight_A[mask[0],:]
                        merge_lora_B_vec = original_weight_B[:,mask[0]]
                        # change 3
                        if merge_percentage == "half":
                            merge_lora_gate_vec = original_weight_gate[:,mask[0]] - self.gate_imp_softmax(p)[:,mask[0]]
                        elif merge_percentage == "full":
                            merge_lora_gate_vec = original_weight_gate[:,mask[0]]

                        # print(merge_lora_A_vec.shape)
                        # print(merge_lora_B_vec.shape)
                        #print((original_weight_A[~mask1] * value_ipt[n][mask]).unsqueeze(0))
                        target.weight.data += (merge_lora_B_vec @ (merge_lora_A_vec * merge_lora_gate_vec.T)) * target.scaling[active_adapter]
            
        def reset_rank_wo_gate(active_adapter):
            # current_param = self.cache_lora_param_state()
            for name, module in self.model.named_modules():
                if hasattr(module, "recorded_activations"):
                    
                    mask = masks[name]
                    if inherit_param:
                        inherit_param_ckpt_wo_gate(value_ipt[name],mask,module,active_adapter,name)
                    if optimizer_states:
                        #print("reset optimizer.")
                        weight_A_state = optimizer_states.state[module.lora_A[active_adapter].weight]
                        weight_B_state = optimizer_states.state[module.lora_B[active_adapter].weight]
                        weight_A_state["exp_avg"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg"][mask[0],:])
                        weight_A_state["exp_avg_sq"][mask[0],:] = torch.zeros_like(weight_A_state["exp_avg_sq"][mask[0],:])
                        weight_B_state["exp_avg"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg"][:,mask[0]])
                        weight_B_state["exp_avg_sq"][:,mask[0]] = torch.zeros_like(weight_B_state["exp_avg_sq"][:,mask[0]])
                             
        with torch.no_grad():
            for active_adapter in self.active_adapters:
                if has_gate:
                    reset_rank_with_gate(active_adapter)
                else:
                    reset_rank_wo_gate(active_adapter)


    def save_activations(self):
        result_ipt = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "recorded_activations"):
                result_ipt[name] = module.recorded_activations.unsqueeze(0)
        return result_ipt
    def save_gate(self):
        result_ipt = {}
        for n, p in self.model.named_parameters():
            if f"lora_gate." in n:
                result_ipt[n] = self.gate_imp_softmax(p).detach()
        # del self.old_state_dict
        # self.old_state_dict = self.cache_lora_param_state() 
        return result_ipt
    def save_CKA(self,directory_path=None): 
        import matplotlib.pyplot as plt
        import seaborn as sns
        def cka(matrix1, matrix2):
            matrix1_centered = matrix1 - matrix1.mean(dim=0)
            matrix2_centered = matrix2 - matrix2.mean(dim=0)

            K_x = F.linear(matrix1_centered, matrix1_centered)
            K_y = F.linear(matrix2_centered, matrix2_centered)

            hsic = torch.trace(torch.mm(K_x, K_y)) / K_x.shape[0]**2
            var1 = torch.trace(torch.mm(K_x, K_x)) / K_x.shape[0]**2
            var2 = torch.trace(torch.mm(K_y, K_y)) / K_y.shape[0]**2

            return hsic / torch.sqrt(var1 * var2)
        with torch.no_grad():
            for active_adapter in self.active_adapters: 
                for n, p in self.model.named_parameters():
                    if f"lora_A." in n:
                        name = n.replace(".lora_A."+active_adapter+".weight","")
                        _, target, _ = _get_submodules(self.model, name)
                        original_weight_A = target.lora_A[active_adapter].weight.data.clone()
                        original_weight_B = target.lora_B[active_adapter].weight.data.clone()
                        r = original_weight_A.shape[0]
                        rank_chunks = []
                        similarity_matrix = torch.zeros((r, r)).to(original_weight_A.device)
                        for i in range(r):
                            rank_chunks.append(original_weight_B[:,[i]] @ original_weight_A[[i],:])
                        start_time = time.time()
                        for i in range(r):
                            for j in range(r):
                                similarity_matrix[i][j] = cka(rank_chunks[i], rank_chunks[j])
                        similarity_matrix_np = similarity_matrix.cpu().numpy()

                        plt.figure(figsize=(5, 4))
                        sns.heatmap(similarity_matrix_np, cmap="coolwarm")
                        plt.title('CKA Similarity Heatmap')
                            
                        plt.savefig(os.path.join(directory_path, f'{name}_heatmap.png'))
                        plt.close()
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Elapsed time: {elapsed_time} seconds")  
    def save_imp(self, imp_criteria=None,global_step=None,output_path=None):
        directory_path = output_path+'/rank_logs'
        try:
            os.mkdir(directory_path)
            print(f"Dictionary created successfully at path '{directory_path}'")
        except FileExistsError:
            print(f"Dictionary already exists at path '{directory_path}'")
        if imp_criteria=="first_order":
            first_order_imp = self.first_order_importance()
            first_order_imp_save = {n: value.tolist() for n, value in first_order_imp.items()}
            with open(directory_path+'/first_order_imp_'+str(global_step)+'.json', 'w') as f:
                json.dump(first_order_imp_save, f)
        if imp_criteria=="activation":
            act_imp = self.save_activations()
            act_imp_save = {n: value.tolist() for n, value in act_imp.items()}
            with open(directory_path+'/act_imp_'+str(global_step)+'.json', 'w') as f:
                json.dump(act_imp_save, f)
        if imp_criteria=="gate":
            gate_imp = self.save_gate()
            gate_imp_save = {n: value.tolist() for n, value in gate_imp.items()}
            with open(directory_path+'/gate_imp_'+str(global_step)+'.json', 'w') as f:
                json.dump(gate_imp_save, f)
        if imp_criteria=="cka":
            directory_path = output_path+"/"+str(global_step)
            try:
                os.mkdir(directory_path)
                print(f"Dictionary created successfully at path '{directory_path}'")
            except FileExistsError:
                print(f"Dictionary already exists at path '{directory_path}'")
            cka_imp = self.save_CKA(directory_path=directory_path)

    def set_activation_hook(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                module.record_activations = True

    def unset_weighted_erank(self):
        for active_adapter in self.active_adapters: 
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    module.use_erank[active_adapter] = False