import os
import sys
from typing import List
from copy import deepcopy
import json
import torch
import transformers
import datasets
from datasets import load_from_disk
import pdb
import copy
import torch.nn.functional as F
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaTokenizer
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.optimization import get_scheduler

from transformers import Trainer

from typing import  Optional
from torch import nn


_is_torch_generator_available = True
from transformers.file_utils import is_datasets_available
from torch.utils.data import RandomSampler, SequentialSampler
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from transformers.trainer_utils import has_length
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
from training_utils import CosineAnnealingThreshold

logger = logging.get_logger(__name__)

SAVE_PATH = ''
save_merged_model_ckpt = False
tokenizer = None
min_k = False
evaluation_method = "gate_value"
merge_on_half_steps = False
inherit_param = False
rank_threshold = 0.2
rank_cumsum_threshold = 0.95
reverse_gate = False
act_temperature = 1.0
reverse_imp = False
decay_threshold = False
end_step=5000
calculate_sim_step = 0
RANDOM_MASK=False
ONLY_PRUNING=False
CONS_K=False
DO_NOTHING=False
class ResetReloraCallback(TrainerCallback):
    def __init__(self, T=50, reset_optimizer=True,
                 relora_warmup_step=50, is_pretrain=False, relora_scheduler=False,
                 remora_types=2, plora_m=0.0):
        self.T = T
        self.reset_optimizer = reset_optimizer
        self.relora_warmup_step = relora_warmup_step
        self.is_pretrain = is_pretrain
        self.relora_scheduler = relora_scheduler
        self.remora_types = remora_types
        self.plora_m = plora_m



    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model'].base_model
        optimizer = kwargs['optimizer']
        use_plora = self.plora_m != 0
        
        if state.global_step % self.T == 0 and state.global_step > 0 and (state.max_steps-state.global_step)>=self.T:
            print("Use PLoRA: ",str(use_plora))
            for layer in  kwargs['model'].base_model.model.model.layers:
                for linear in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj,
                               layer.mlp.gate_proj, layer.mlp.down_proj, layer.mlp.up_proj]:
                    linear.merge(use_plora=use_plora)
                    print(linear.merged_adapters)
                    linear.merged_adapters = []
                    if linear.use_mora['default']:
                        #print('mora change type', linear.mora_type['default'], end='->')
                        if self.remora_types == 4:
                            # 1->2->3->4
                            mora_type_map = {1:2, 2:3, 3:4, 4:1}
                        else:
                            mora_type_map = {1:2, 2:1}

                        print('mora change type', linear.mora_type['default'], end='->')
                        print(mora_type_map[linear.mora_type['default']])
                        linear.reset_lora_parameters('default', init_lora_weights=True,
                                                     mora_type=mora_type_map[linear.mora_type['default']])
                        #print(linear.mora_type['default'])
                    else:
                        if not use_plora:
                            print("reset param.")
                            linear.reset_lora_parameters('default', init_lora_weights=True)
                        else:
                            print("Not Reset param because plora_m is",str(self.plora_m))
            # save base model
            if save_merged_model_ckpt:
                print('save base model', os.path.join(SAVE_PATH, "base-model"), self.reset_optimizer)
                save_model = deepcopy(kwargs['model'])
                save_model.merge_and_unload().save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                tokenizer.save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                del save_model

            if self.reset_optimizer:
                # reset optimizer
                from collections import defaultdict
                #optimizer.__setstate__({'state': defaultdict(dict)})
                if self.is_pretrain:
                    for name, param in model.named_parameters():
                        if 'lora' in name:
                            del optimizer.state[param]
                else:
                    optimizer.state = defaultdict(dict)

                if (not self.relora_scheduler) and self.plora_m == 0:
                    # if we use relora scheduler, we don't need to reset scheduler
                    # reset warmup steps to 50
                    scheduler = kwargs['lr_scheduler']
                    part =  scheduler.lr_lambdas[0]
                    _,_, f = part.__reduce__()
                    f, _, k, n = f
                    k['num_warmup_steps'] = self.relora_warmup_step
                    k['num_training_steps'] = state.max_steps-state.global_step

                    scheduler._step_count = 0
                    scheduler.last_epoch = 0

                    scheduler._step_count = 0
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] = scheduler._last_lr[0]
                    print('reset scheduler', scheduler._last_lr[0], scheduler.state_dict())
            else:
                print('not reset optimizer')

class DynamicRankCallback(TrainerCallback):
    def __init__(self, use_adalora, use_beamlora, T, merge_to_base_model, reset_optimizer, reset_param):
        self.use_adalora = use_adalora
        self.use_beamlora = use_beamlora
        self.T = T
        self.ckpt_T = self.T/2
        self.is_final_reset = False
        self.merge_to_base_model = merge_to_base_model
        self.reset_optimizer = reset_optimizer
        self.reset_param = reset_param
        if self.use_beamlora:
            print("end step:",str(end_step))
            self.p_threshold_scheduler = CosineAnnealingThreshold(T_max=end_step,eta_max=1-rank_cumsum_threshold)
    # def on_step_begin(self, args, state, control, **kwargs):
    #     model = kwargs['model']
    #     for name,param in model.named_parameters():
    #         if 'lora' in name:
    #             print(param)
    #             break

        
    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.use_adalora:
            #print("Use AdaLoRA")
            model = kwargs['model']
            model.base_model.update_and_allocate(state.global_step)
        if self.use_beamlora:
            model = kwargs['model']
            optimizer = kwargs['optimizer']
            if state.global_step % self.T != 0 and state.global_step % self.ckpt_T == 0 and state.global_step > 0:
                print("cache state")
                model.base_model.cache_state(optimizer_state=optimizer)
                    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        if calculate_sim_step!=0:
            if state.global_step > self.T and state.global_step % calculate_sim_step == 0:
                print("Calculate Sim")
                model.base_model.calculate_rank_similarity(global_step=state.global_step,directory_path=SAVE_PATH)
        if state.global_step % self.T == 0 and state.global_step > 0 and (state.max_steps-state.global_step)>=self.T:
            if self.use_beamlora and not DO_NOTHING:
                print("Use ELoRA")
                if merge_on_half_steps and (state.global_step>(state.max_steps/2)):
                    print("Merge on half step Done!")
                else:
                    # if not self.merge_to_base_model:
                    #     model.save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                    # else:
                    #     print('save base model', os.path.join(SAVE_PATH, "base-model"))
                    #     save_model = deepcopy(kwargs['model'])
                    #     save_model.merge_and_unload().save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                    #     tokenizer.save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                    #     del save_model

                    # if not self.is_final_reset:
                    #     self.is_final_reset = (state.max_steps-state.global_step)< 4*self.T
                    #     model.base_model.reset_rank(threshold=0.2,global_step=state.global_step,reset_rank=True,is_final_reset=self.is_final_reset)
                    if inherit_param:
                        print("use_inherit_param")
                        global rank_threshold
                        masks, value_ipt = model.base_model.analyze_rank_importance(threshold=rank_threshold,
                                                                                cumsum_threshold=1-self.p_threshold_scheduler.get_threshold(),
                                                                                min_k=True,
                                                                                global_step=state.global_step,
                                                                                evaluation_method=evaluation_method,
                                                                                optimizer_states=optimizer,
                                                                                output_path=SAVE_PATH,
                                                                                random_mask=RANDOM_MASK,
                                                                                constant_K=CONS_K)
                        model.base_model.reset_rank(merge_to_base_model=self.merge_to_base_model,
                                                    optimizer_states=optimizer if self.reset_optimizer else None,
                                                    masks=masks,
                                                    inherit_param=inherit_param,
                                                    reset_param=self.reset_param,
                                                    only_pruning=ONLY_PRUNING,
                                                    random_mask=RANDOM_MASK,)
                        model.base_model.save_imp(global_step=state.global_step,imp_criteria="gate",output_path=SAVE_PATH)
                        print("Calculate Sim")
                        if calculate_sim_step!=0:
                            model.base_model.calculate_rank_similarity(global_step=state.global_step,directory_path=SAVE_PATH)
                        if decay_threshold:
                            print("use_decay_threshold")
                            rank_threshold /= 2
                    elif reverse_gate:
                        print("use_reverse_gate")
                        masks, value_ipt = model.base_model.analyze_rank_importance(threshold=rank_threshold,
                                                                                min_k=True,
                                                                                global_step=state.global_step,
                                                                                evaluation_method=evaluation_method,
                                                                                optimizer_states=optimizer,
                                                                                output_path=SAVE_PATH)
                        model.base_model.reset_rank(merge_to_base_model=self.merge_to_base_model,
                                                    optimizer_states=optimizer if self.reset_optimizer else None,
                                                    masks=masks,
                                                    reverse_gate=reverse_gate,
                                                    inherit_param=inherit_param,
                                                    reset_param=self.reset_param)                        
                    else:
                        masks, value_ipt = model.base_model.analyze_rank_importance(threshold=rank_threshold,
                                                                                    min_k=min_k,
                                                                                    global_step=state.global_step,
                                                                                    evaluation_method=evaluation_method,
                                                                                    optimizer_states=optimizer,
                                                                                    output_path=SAVE_PATH)
                        model.base_model.reset_rank(merge_to_base_model=self.merge_to_base_model,
                                                    optimizer_states=optimizer if self.reset_optimizer else None,
                                                    masks=masks,
                                                    reset_param=self.reset_param)
        if self.use_beamlora:
            self.p_threshold_scheduler.step()
    def on_prediction_step(self, args, state, control, **kwargs):
        import pdb
        pdb.set_trace()

class AnalyzeRankCallback(TrainerCallback):
    def __init__(self, T, imp_criteria):
        self.imp_criteria = imp_criteria
        self.T = T

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.imp_criteria=="first_order":
            model = kwargs['model']
            model.base_model.update_ipt()

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        if state.global_step % self.T == 0 and state.global_step > 0:
            print(self.imp_criteria)
            model.base_model.save_imp(global_step=state.global_step,imp_criteria=self.imp_criteria,output_path=SAVE_PATH)
    def on_prediction_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        if state.global_eval_step % self.T == 0 and state.global_eval_step > 0:
            print(self.imp_criteria)
            model.base_model.save_imp(global_step=state.global_eval_step,imp_criteria=self.imp_criteria,output_path=SAVE_PATH)

class LinearOutputNormTrackerCallback(TrainerCallback):
    def __init__(self, T, tracker):
        self.tracker = tracker
        self.T = T
        self.directory_path = SAVE_PATH+'/rank_logs'
        try:
            os.mkdir(self.directory_path)
            print(f"Folder '{self.directory_path}' is created")
        except FileExistsError:
            print(f"Folder '{self.directory_path}' already exists")
    def on_prediction_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        if state.global_eval_step % self.T == 0 and state.global_eval_step > 0:
            act_imp = self.tracker.get_output_norms()
            act_imp_save = {n: value.tolist() for n, value in act_imp.items()}
            with open(self.directory_path+'/act_imp_'+str(state.global_eval_step)+'.json', 'w') as f:
                json.dump(act_imp_save, f)

class ElasticRankCallback(TrainerCallback):
    def __init__(self, T, merge_to_base_model, reset_optimizer, reset_param):
        self.T = T
        self.is_final_reset = False
        self.merge_to_base_model = merge_to_base_model
        self.reset_optimizer = reset_optimizer
        self.reset_param = reset_param
    # def on_step_begin(self, args, state, control, **kwargs):
    #     model = kwargs['model']
    #     for name,param in model.named_parameters():
    #         if 'lora' in name:
    #             print(param)
    #             break
    # def on_pre_optimizer_step(self, args, state, control, **kwargs):
    #     model = kwargs['model']
    #     for name,param in model.named_parameters():
    #         if 'lora' in name:
    #             print(param)
    #             break
    # def on_optimizer_step(self, args, state, control, **kwargs):
    #     if self.use_adalora:
    #         #print("Use AdaLoRA")
    #         model = kwargs['model']
    #         model.base_model.update_and_allocate(state.global_step)
    #     if self.use_beamlora:
    #         model = kwargs['model']
    #         optimizer = kwargs['optimizer']
            
            # for name, param in model.named_parameters():
            #     if 'layers.0.self_attn.v_proj.lora_gate' in name:
            #         import pdb
            #         pdb.set_trace()
                    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        optimizer = kwargs['optimizer']
        if state.global_step % self.T == 0 and state.global_step > 0 and (state.max_steps-state.global_step)>=self.T:
            print("Reset Rank!")
            if merge_on_half_steps and (state.global_step>(state.max_steps/2)):
                print("Merge on half step Done!")
                model.unset_weighted_erank()
            else:
                # if not self.merge_to_base_model:
                #     model.save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                # else:
                #     print('save base model', os.path.join(SAVE_PATH, "base-model"))
                #     save_model = deepcopy(kwargs['model'])
                #     save_model.merge_and_unload().save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                #     tokenizer.save_pretrained(os.path.join(SAVE_PATH, str(state.global_step)+"-model"))
                #     del save_model

                # if not self.is_final_reset:
                #     self.is_final_reset = (state.max_steps-state.global_step)< 4*self.T
                #     model.base_model.reset_rank(threshold=0.2,global_step=state.global_step,reset_rank=True,is_final_reset=self.is_final_reset)
                if inherit_param:
                    print(rank_threshold)
                    masks, value_ipt = model.base_model.analyze_rank_importance(threshold=rank_threshold,
                                                                            min_k=True,
                                                                            global_step=state.global_step,
                                                                            evaluation_method=evaluation_method,
                                                                            optimizer_states=optimizer,
                                                                            output_path=SAVE_PATH,
                                                                            activation_temperature=act_temperature,
                                                                            reverse_imp=reverse_imp)
                    model.base_model.reset_rank(merge_to_base_model=self.merge_to_base_model,
                                                value_ipt=value_ipt,
                                                optimizer_states=optimizer if self.reset_optimizer else None,
                                                masks=masks,
                                                inherit_param=inherit_param,
                                                reset_param=self.reset_param,
                                                has_gate=False)
        elif (state.max_steps-state.global_step)<self.T:
            model.unset_weighted_erank()


    def on_prediction_step(self, args, state, control, **kwargs):
        import pdb
        pdb.set_trace()

class OurTrainer(Trainer):
    shuffle_data = True
    lora_plus_lambda = 1
    use_relora = False
    use_relora_step = 50
    reset_optimizer = True
    reset_param = False
    relora_warmup_step = 50
    is_pretrain = False
    relora_scheduler = False
    remora_types = 2
    merge_to_base_model = False
    plora_m = 0.0
    use_adalora = False
    use_beamlora = False
    use_beamlora_step = 100
    imp_criteria = None
    use_erank = False


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            lora_gate_params = set([n for n, p in opt_model.named_parameters() if 'lora_gate' in n])
            if self.lora_plus_lambda > 1:
                lora_b_params = set([n for n, p in opt_model.named_parameters() if 'lora_B' in n])
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in lora_b_params)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in lora_b_params)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate * self.lora_plus_lambda,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ]
                print(len(optimizer_grouped_parameters[0]['params']), len(optimizer_grouped_parameters[1]['params']), len(optimizer_grouped_parameters[2]['params']))
            elif len(lora_gate_params)>0:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in lora_gate_params)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in lora_gate_params)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.use_relora:
            self.add_callback(ResetReloraCallback(T=self.use_relora_step,
                                                 reset_optimizer=self.reset_optimizer,
                                                #  reset_param=self.reset_param,
                                                 relora_warmup_step=self.relora_warmup_step,
                                                 is_pretrain=self.is_pretrain,
                                                 relora_scheduler=self.relora_scheduler,
                                                 remora_types=self.remora_types,
                                                 plora_m = self.plora_m)
                                                 )
        if self.use_adalora or self.use_beamlora:
            self.add_callback(DynamicRankCallback(use_adalora=self.use_adalora,use_beamlora=self.use_beamlora,T=self.use_beamlora_step,merge_to_base_model=self.merge_to_base_model,
            reset_optimizer=self.reset_optimizer,reset_param=self.reset_param))
        if self.imp_criteria is not None:
            self.add_callback(AnalyzeRankCallback(T=500, imp_criteria=self.imp_criteria))
        if self.use_erank:
            self.add_callback(ElasticRankCallback(T=self.use_beamlora_step,merge_to_base_model=self.merge_to_base_model,
            reset_optimizer=self.reset_optimizer,reset_param=self.reset_param))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        elif not self.shuffle_data:
            return SequentialSampler(self.train_dataset)
        else:
            return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if 'labels' not in inputs:
            inputs['labels'] = inputs['input_ids'].clone()

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # sync loss from all processes
        self.state.loss = self._nested_gather(loss).mean().item()

        #  model.base_model.model.model.layers[0].mlp.gate_proj.lora_A['default'].weight
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.relora_scheduler:
                from training_utils import get_scheculer as relora_get_scheduler
                if num_training_steps % self.use_relora_step > 0:
                    num_training_steps = ((num_training_steps // self.use_relora_step)+1)*self.use_relora_step
                self.lr_scheduler = relora_get_scheduler(
                    scheduler_type='cosine_restarts',
                    optimizer=optimizer,
                    num_training_steps=num_training_steps,
                    warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    min_lr_ratio=0.1,
                    cycle_length=self.use_relora_step,
                    restart_warmup_steps=self.relora_warmup_step,
                    adjust_step=0,
                )

            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        if self.use_adalora:
            self.model.base_model.peft_config["default"].total_step = num_training_steps
            print("total step :", str(num_training_steps))
        return self.lr_scheduler


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "",
        dataset_split: str = "",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 1,
        learning_rate: float = 3e-4,
        lr_scheduler_type: str = 'linear',
        cutoff_len: int = 2048,
        val_set_size: int = 0,
        ddp_find_unused_parameters: bool = False,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"
        ],
        # llm hyperparams
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        seed: int = 42,
        use_4bit: bool = False,
        use_16bit: bool = False,
        debug: bool = False,
        full_ft: bool = False,
        deepspeed: str = None,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        use_flash_atten: bool = False,
        not_shuffle_data: bool = False,
        max_steps: int = -1,
        use_gptq: bool = False,
        use_bf16: bool = False,
        train_embhead: bool = False,
        max_samples: int = -1,
        save_total_limit: int = 7,
        new_pad_token: bool = False,
        save_steps: int = 200,
        grad_checkpoint: bool = False,
        pretrain: str = None,
        #log
        imp_criteria: str = None,
        get_linear_output: bool = False,
        # erank
        use_erank: bool = False,
        use_weighted_erank: bool = False,
        activation_temperature: int = 1,
        use_reverse_imp=False,
        cumsum_threshold=0.95,
        calculate_rank_sim_step=0,
        # dora
        use_dora: bool = False,
        # pissa
        use_pissa: bool = False,
        # alora
        use_alora: bool = False,
        # elora
        only_add_gate: bool = False,
        use_beamlora: bool = False,
        elora_temperature: float = 0.05,
        use_beamlora_step: int = 100,
        merge_to_base_model: bool = False,
        use_min_k: bool = False,
        reset_optimizer: bool = False,
        reset_param: bool = False,
        select_evaluation_method: str = "gate_value",
        only_merge_on_half_steps: bool = False,
        use_inherit_param: bool = False,
        threshold: float = 0.2,
        use_reverse_gate=False,
        infer_on_train_dataset: bool = False,
        saved_peft_path: str = None,
        use_decay_threshold: bool = False,
        reset_end_step: int = 12344,
        # lora+
        lora_plus_lambda: int = 1,
        # adalora
        use_increlora: bool = False,
        use_adalora: bool = False,
        init_r: int = 64,
        target_r: int = 32,
        beta1: float = 0.85,
        beta2: float = 0.85,
        tinit: int = 200,
        tfinal: int = 1000,
        deltaT: int = 10,
        orth_reg_weight: float = 0.1,
        # asylora
        use_asymmetriclora: bool = False,
        # mora
        use_mora: bool = False,
        mora_type: int = 1,
        # relora
        use_relora: bool = False,
        use_relora_step: int = 50,
        use_relora_not_reset_optimizer: bool = False ,
        relora_warmup_step: int = 50,
        relora_scheduler: bool = False,
        remora_types: int = 4,
        plora_m: float = 0.0,
        save_merged_model_checkpoint: bool = False,
        #ablation
        random_mask: bool = False,
        only_pruning: bool = False,
        constant_K: bool = False,
        do_nothing: bool = False,
        #transfer
        knowledge_masking: float = 0.0,
        knowledge_dropping: float = 0.0,
):
    global SAVE_PATH
    global save_merged_model_ckpt
    global tokenizer
    global min_k
    global evaluation_method
    global merge_on_half_steps
    global inherit_param
    global rank_threshold
    global reverse_gate
    global act_temperature
    global reverse_imp
    global decay_threshold
    global rank_cumsum_threshold
    global end_step
    global RANDOM_MASK
    global ONLY_PRUNING
    global CONS_K
    global DO_NOTHING
    global calculate_sim_step
    calculate_sim_step = calculate_rank_sim_step
    DO_NOTHING = do_nothing
    RANDOM_MASK = random_mask
    ONLY_PRUNING = only_pruning
    CONS_K = constant_K
    end_step = reset_end_step
    decay_threshold = use_decay_threshold
    reverse_imp = use_reverse_imp
    act_temperature = activation_temperature
    reverse_gate = use_reverse_gate
    rank_threshold = threshold
    rank_cumsum_threshold = cumsum_threshold
    inherit_param = use_inherit_param
    merge_on_half_steps = only_merge_on_half_steps
    evaluation_method = select_evaluation_method
    min_k = use_min_k
    save_merged_model_ckpt = save_merged_model_checkpoint
    print("save_merged_model_checkpoint:",str(save_merged_model_ckpt))
    set_seed(seed)
    gradient_accumulation_steps = batch_size // micro_batch_size

    output_dir = wandb_run_name

    # bug in transformers
    if output_dir == wandb_run_name:
        if "Mistral" in base_model:
            if "cs" in wandb_run_name:
                output_dir = 'cs_ckpts-mistral/' + output_dir
            elif "code" in wandb_run_name:
                output_dir = 'code_ckpts-mistral/' + output_dir
            else:
                output_dir = 'ckpts-mistral/' + output_dir
        else:
            if "cs" in wandb_run_name:
                output_dir = 'cs_ckpts/' + output_dir
            elif "code" in wandb_run_name:
                output_dir = 'code_ckpts/' + output_dir
            else:
                output_dir = 'ckpts/' + output_dir
        

    SAVE_PATH = output_dir

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        #torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)
    else:
        rank = 0

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    MODEL_CLASS = AutoModelForCausalLM

    if debug:
        # random init
        config = AutoConfig.from_pretrained(base_model)
        config.num_hidden_layers = 1
        model = MODEL_CLASS(config)
        use_wandb = False
    elif pretrain == '250m':
        config = AutoConfig.from_pretrained('./configs/llama_250m.json')
        model = MODEL_CLASS(config)
    elif pretrain == '1b':
        config = AutoConfig.from_pretrained('./configs/llama_1b.json')
        model = MODEL_CLASS(config)
    elif use_4bit:
        from transformers import BitsAndBytesConfig
        model = MODEL_CLASS.from_pretrained(
            base_model,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            device_map=device_map,
        )
    else:
        from transformers import BitsAndBytesConfig
        torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
        config=None
        if knowledge_masking!=0 or knowledge_dropping!=0:
            config = AutoConfig.from_pretrained(base_model,trust_remote_code=True)
            config.knowledge_masking=knowledge_masking
            config.knowledge_dropping=knowledge_dropping
            print(config)
        model = MODEL_CLASS.from_pretrained(
            base_model,
            load_in_8bit=False if full_ft or (deepspeed and 'ds3' in deepspeed) or use_16bit else True, # if use zero3 not quantize
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_flash_attention_2=use_flash_atten,
            config=config,
            trust_remote_code=True
        )


    if pretrain is not None:
        print('saving init model')
        if rank == 0:
            model.save_pretrained(os.path.join(SAVE_PATH, "init-model"))


    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)

    if new_pad_token:
        import deepspeed as dsp
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token_id = (
        # NOTE: set this to eos token, set to unk(0) while make output nan
            2  # unk. we want this to be different from the eos token
        )
        tokenizer =  AutoTokenizer.from_pretrained(base_model, use_fast=False)
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        embeddings = model.get_input_embeddings()
        with dsp.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
            embedding_size = embeddings.weight.shape[0]
            if len(tokenizer) > embeddings.weight.shape[0]:
                model.resize_token_embeddings(len(tokenizer))
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest",
        )
    else:
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        )

    if full_ft:
        print('not use peft')
    else:
        if grad_checkpoint:
            model.enable_input_require_grads()

        if (deepspeed and 'ds3' not in deepspeed) and not use_16bit:
            model = prepare_model_for_kbit_training(model)

        # 'q_proj k_proj v_proj o_proj gate_proj down_proj up_proj'
        if type(lora_target_modules) is str:
            lora_target_modules = [lora_target_modules]
        CONFIGCLASS = LoraConfig

        if use_adalora:
            from peft import AdaLoraConfig
            CONFIGCLASS = AdaLoraConfig

        kwargs = {}
        kwargs['use_dora'] = use_dora
        kwargs['use_pissa'] = use_pissa
        if use_mora:
            kwargs['use_mora'] = True
            kwargs['mora_type'] = mora_type
            print('mora type', mora_type)
        if plora_m != 0:
            kwargs['plora_m'] = plora_m
            print(plora_m)
        kwargs['use_beamlora'] = use_beamlora
        if use_beamlora:
            kwargs['elora_temperature'] = elora_temperature
        kwargs['only_add_gate'] = only_add_gate
        kwargs['use_erank'] = use_weighted_erank
        if imp_criteria=="activation":
            print("activation")
            kwargs['record_activations'] = True
        if train_embhead:
            kwargs['modules_to_save'] = ['embed_tokens', 'lm_head', 'norm', 'input_layernorm', 'post_attention_layernorm' ]
        if not use_adalora:
            config = CONFIGCLASS(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear" if "all-linear" in str(lora_target_modules) else lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                **kwargs,
            )
        else:
            config = CONFIGCLASS(
                init_r=init_r,
                target_r=target_r,
                beta1=beta1,
                beta2=beta2,
                tinit=tinit,
                tfinal=tfinal,
                deltaT=deltaT,
                orth_reg_weight=orth_reg_weight,
                lora_alpha=lora_alpha,
                target_modules="all-linear" if "all-linear" in str(lora_target_modules) else lora_target_modules,
                lora_dropout=lora_dropout,
                use_increlora=use_increlora,
                bias="none",
                task_type="CAUSAL_LM",
                **kwargs,
            )
        if saved_peft_path is not None:
            model = PeftModel.from_pretrained(model,saved_peft_path)
            if imp_criteria=="activations":
                model.set_activation_hook() 
        else:
            model = get_peft_model(model, config)

        if use_4bit:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16 if use_bf16 else torch.float16)
                    #module = module.to(torch.float32)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        module = module.to(torch.bfloat16 if use_bf16 else torch.float16)

    if not full_ft:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if use_asymmetriclora:
        from tqdm import tqdm
        bar = tqdm(total=len([n for n, p in model.named_parameters() if 'lora_A' in n]))
        asy_dict = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                shape = param.shape
                random_w = torch.rand(shape[1], max(shape[1], 4096)).cuda()
                # slow here
                U_rand, S_rand, V_rand = torch.linalg.svd(random_w)
                print(name, shape, V_rand.std().item(), V_rand.mean().item())
                param.data = V_rand[:, :shape[0]].T.contiguous()
                asy_dict[name] = param.data.clone().cpu()
                bar.update(1)
                #param.requires_grad = False
            #elif 'lora_B' in name:
                #param.requires_grad = True
        bar.close()

    if max_samples > 0:
        print(f'use max samples {max_samples}')
        train_data = train_data.shuffle(seed=42)
        train_data = train_data.select(range(max_samples))

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    warmup_ratio = 0
    val_data = None
    if 'meta-math' in data_path:
        class A:
            pass
        data_args = A()
        data_args.data_path =  'meta-math/MetaMathQA'
        data_args.data_length = 1000000
        data_args.dataset_split = dataset_split
        from training_utils import make_supervised_data_module
        lr_scheduler_type = lr_scheduler_type
        print(lr_scheduler_type)
        save_steps = save_steps if not save_merged_model_ckpt else 1e10
        print("Save Steps:",str(save_steps))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            model_max_length=512,
            #padding_side="right",
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        #tokenizer.pad_token = "[PAD]"
        #tokenizer.padding_side = "left"
        tokenizer.pad_token_id = (
        # NOTE: set this to eos token, set to unk(0) while make output nan
            2  # unk. we want this to be different from the eos token
        )
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        train_data = data_module['train_dataset']
        data_collator = data_module['data_collator']
        warmup_steps, warmup_ratio = 0, 0.03
    elif 'Code' in data_path:
        class A:
            pass
        data_args = A()
        data_args.data_path =  'fxmeng/CodeFeedback-Python105K'
        #data_args.data_length = 1000000
        data_args.dataset_split = dataset_split
        from training_utils import make_supervised_data_module
        lr_scheduler_type = lr_scheduler_type
        print(lr_scheduler_type)
        save_steps = save_steps if not save_merged_model_ckpt else 1e10
        print("Save Steps:",str(save_steps))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            model_max_length=1024,
            #padding_side="right",
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        #tokenizer.pad_token = "[PAD]"
        #tokenizer.padding_side = "left"
        tokenizer.pad_token_id = (
        # NOTE: set this to eos token, set to unk(0) while make output nan
            2  # unk. we want this to be different from the eos token
        )
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        train_data = data_module['train_dataset']
        data_collator = data_module['data_collator']
        warmup_steps, warmup_ratio = 0, 0.03
    elif 'commonsense' in data_path:
        from training_utils import load_commonsense_datasets,print_trainable_parameters_and_names
        print_trainable_parameters_and_names(model)
        warmup_steps = 100
        lr_scheduler_type = 'linear'
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            model_max_length=512,
            padding_side="left",# Allow batched inference
            use_fast=False,
            trust_remote_code=True
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        train_data, val_data = load_commonsense_datasets(data_path,tokenizer)
        # change lora to float  
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()


    else:
        train_data = load_from_disk(data_path)
        if 'open-instruct-tokenized' in data_path:
            prev_len = len(train_data)
            #train_data = train_data.filter(lambda x: max(x['input_ids']) < 32000,num_proc=48)
            def remap(entry):
                entry['input_ids'] = [x if x < 32000 else 0 for x in entry['input_ids']]
                return entry
            # this sample contain <pad> which is add new token in prev
            print(f'filter out {prev_len - len(train_data)} samples')
            if cutoff_len != 2048:
                def cut_off(entry):
                    entry['input_ids'] = entry['input_ids'][:cutoff_len]
                    entry['attention_mask'] = entry['attention_mask'][:cutoff_len]
                    entry['labels'] = entry['labels'][:cutoff_len]
                    return entry
                train_data = train_data.map(cut_off, num_proc=48)
                train_data = train_data.filter(lambda example: (torch.LongTensor(example['labels']) != -100).any(), num_proc=48)

    if infer_on_train_dataset:
        val_data = copy.deepcopy(train_data)
        if get_linear_output:
            model = model.merge_and_unload()
            from training_utils import LinearOutputNormTracker
            tracker = LinearOutputNormTracker(model)
            tracker.add_hooks()
            
    TRAINER_CLS = OurTrainer
    trainer = TRAINER_CLS(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            #max_steps=10000,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            fp16=False if use_bf16 else True,
            bf16=use_bf16,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=ddp_find_unused_parameters if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
            deepspeed=deepspeed,
            seed=seed,
            gradient_checkpointing=grad_checkpoint,
            fsdp='full_shard auto_wrap' if full_ft and not deepspeed and pretrain is None else '',
            fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer' if full_ft and not deepspeed and pretrain is None else None,
        ),
        data_collator=data_collator,
    )
    trainer.lora_plus_lambda = lora_plus_lambda
    trainer.use_relora = use_relora
    trainer.use_relora_step = use_relora_step
    if use_relora:
        trainer.reset_optimizer = not use_relora_not_reset_optimizer
    else:
        trainer.reset_optimizer = reset_optimizer
    trainer.reset_param = reset_param
    trainer.relora_warmup_step = relora_warmup_step
    trainer.is_pretrain = pretrain is not None
    trainer.relora_scheduler = relora_scheduler
    trainer.remora_types = remora_types
    trainer.plora_m = plora_m
    trainer.use_adalora = use_adalora
    trainer.use_beamlora = use_beamlora
    trainer.use_beamlora_step = use_beamlora_step
    trainer.merge_to_base_model = merge_to_base_model
    trainer.imp_criteria = imp_criteria
    trainer.use_erank = use_erank

    if not_shuffle_data:
        trainer.shuffle_data = False

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if ddp:
        torch.distributed.barrier()
    if infer_on_train_dataset:
        if trainer.imp_criteria is not None:
            trainer.add_callback(AnalyzeRankCallback(T=500, imp_criteria=imp_criteria))
        if get_linear_output:
            trainer.add_callback(LinearOutputNormTrackerCallback(T=500, tracker=tracker))
        trainer.evaluate()
        return
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if rank == 0:
        if use_relora or use_pissa or merge_to_base_model or use_increlora:
            model.merge_and_unload().save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir)
            if full_ft:
                tokenizer.save_pretrained(output_dir)
            
    if ddp:
        torch.distributed.barrier()


if __name__ == "__main__":
    import fire
    fire.Fire(train)
