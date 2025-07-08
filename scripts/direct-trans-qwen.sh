GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
ABS_PATH="path_to_base_model_folder"
BASE_MODEL="Qwen2-7B"
lr_scheduler_type="cosine"
batch_size=16
RANK=64
echo $RANK
LORA_ALPHA=$((RANK*2))
echo $LORA_ALPHA

SAVE_PROJ=${BASE_MODEL}"-lora_math_direct_trans_r"${RANK}"_transfer_100K"

torchrun   --nproc_per_node=4 --master_port=8885 train.py \
           --base_model ${ABS_PATH}/${BASE_MODEL} --micro_batch_size 4 \
            --wandb_run_name ${SAVE_PROJ} --lora_target_modules q_proj,k_proj,v_proj,o_proj \
            --num_epochs 3 --wandb_project transfer-math --batch_size ${batch_size} \
            --lora_r ${RANK} --lora_alpha ${LORA_ALPHA} --lora_dropout 0.05 \
            --data_path meta-math/MetaMath \
            --seed 42 \
            --save_steps 5000 --save_total_limit 3 \
            --learning_rate 3e-4 \
            --logging_steps 5  --use_bf16  --use_16bit --lr_scheduler_type ${lr_scheduler_type} \
            --dataset_split "train[:100000]"

cd math_infer
source /root/miniconda3/bin/activate math_infer
sleep 1

SAVE_PATH="../ckpts/"${SAVE_PROJ}


CUDA_VISIBLE_DEVICES=$GPU_IDS python gsm8k_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/gsm8k_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 240
CUDA_VISIBLE_DEVICES=$GPU_IDS python math_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/MATH_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 200
BASE_MODEL="Qwen2.5-7B"
CUDA_VISIBLE_DEVICES=$GPU_IDS python gsm8k_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/gsm8k_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 240
CUDA_VISIBLE_DEVICES=$GPU_IDS python math_infer.py --model ${ABS_PATH}/${BASE_MODEL} \
--data_file 'data/MATH_test.jsonl' \
--peft_id $SAVE_PATH \
--tensor_parallel_size 4 \
--batch_size 200