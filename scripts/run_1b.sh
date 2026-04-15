# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=5e-4
export NORM_TYPE=$norm_type
export POST_NUM=$2
export WANDB_MODE=offline
initialization_method='gaussian_FFN_SFLI_g' 
gpu="2,4,6,7"

# Function to run a single training task

echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node 4 --master_port=29500 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr $learning_rates \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "1b_res_${norm_type}_lr${learning_rates}_initialization${initialization_method}_pre_alllayer_100_g_SFLI_Gaussian_SFLI_W_hdcoeffs0.003_bias0.002_multiplegpu_c4" \
    --save_dir "1b_res_${norm_type}_lr${learning_rates}_initialization${initialization_method}_pre_alllayer_100_g_SFLI_Gaussian_SFLI_W_hdcoeffs0.003_bias0.002_multiplegpu_c4" \
    --initialization "$initialization_method" \
    --ffn_bias_step 0.002 \
    --ffn_hd_coeffs 0.003
