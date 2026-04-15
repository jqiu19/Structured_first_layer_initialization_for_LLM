# Define the set of learning rates and normalization types
norm_type=$1
learning_rates=1e-3
initialization_method='gaussian_FFN_SFLI_g'
gpu="3"

export NORM_TYPE="$norm_type"
export POST_NUM="$2"
export WANDB_MODE=offline


echo "Training with learning rate: $learning_rates, norm type: $norm_type on GPU $gpu"

CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node 1 --master_port=29501 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr "$learning_rates" \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --grad_clipping 0.0 \
    --run_name "130m_res_${norm_type}_lr${learning_rates}_postpre3_alllayer_gaussian_gate_only_SFLI_Gaussian_SFLI_hdcoeffs0.003_bias0.002_basestd1.0_multiplegpu_c4" \
    --save_dir "130m_res_${norm_type}_lr${learning_rates}_postpre3_alllayer_gaussian_gate_only_SFLI_Gaussian_SFLI_hdcoeffs0.003_bias0.002_basestd1.0_multiplegpu_c4" \
    --initialization $initialization_method \
    --ffn_bias_step 0.002 \
    --ffn_hd_coeffs 0.003

    #pre_alllayer_gaussian_gate_only_SFLI_Gaussian_SFLI_hdcoeffs0.003_bias0.002_basestd1.0_multiplegpu_c4

    


