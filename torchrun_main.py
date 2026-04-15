import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from initial_scheme import initial_embedding

import bitsandbytes as bnb

import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="default")
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # current-version extra args
    parser.add_argument("--initialization", type=str, default="Xavier", help="how to initialize the model")
    parser.add_argument("--init_name", type=str, default="qr", help="name for the erank_implicit initial embedding scheme")
    parser.add_argument("--sinc_scale", type=float, default=1e2)
    parser.add_argument("--sinc_lamb", type=float, default=1e2)
    parser.add_argument("--qkv_act", type=str, default="cos",
                        choices=["frac_jacobi", "none", "tanh", "sinc", "cos", "hat", "silu", "sin", "cheb", "sinc_poly"],
                        help="Activation applied to Q/K/V after projection: Q=act(UWq), etc.")
    parser.add_argument("--q_hd_coeffs", type=float, default=0.5 * (768 ** (1 / 768) - 1) / 10)
    parser.add_argument("--q_qkv_std", type=float, default=1.0)
    parser.add_argument("--q_bias_start", type=float, default=0.0)
    parser.add_argument("--q_bias_step", type=float, default=1 / 768)
    parser.add_argument("--q_bias_noise_std", type=float, default=0.002)
    parser.add_argument("--k_hd_coeffs", type=float, default=0.5 * (768 ** (1 / 768) - 1) / 10)
    parser.add_argument("--k_qkv_std", type=float, default=1.0)
    parser.add_argument("--k_bias_start", type=float, default=0.0)
    parser.add_argument("--k_bias_step", type=float, default=1 / 768)
    parser.add_argument("--k_bias_noise_std", type=float, default=0.002)
    parser.add_argument("--v_hd_coeffs", type=float, default=0.5 * (768 ** (1 / 768) - 1) / 10)
    parser.add_argument("--v_qkv_std", type=float, default=1.0)
    parser.add_argument("--v_bias_start", type=float, default=0.0)
    parser.add_argument("--v_bias_step", type=float, default=1 / 768)
    parser.add_argument("--v_bias_noise_std", type=float, default=0.002)
    parser.add_argument("--ffn_base_std", type=float, default=1.0)
    parser.add_argument("--ffn_bias_start", type=float, default=0.0)
    parser.add_argument("--ffn_bias_step", type=float, default=1 / 768)
    parser.add_argument("--ffn_bias_noise_std", type=float, default=0.002)
    parser.add_argument("--ffn_hd_coeffs", type=float, default=0.5 * (768 ** (1 / 768) - 1) / 10)
    parser.add_argument("--ffn_act", type=str, default="cos",
                        choices=["frac_jacobi", "none", "tanh", "sinc", "cos", "hat", "silu", "sin", "cheb", "sinc_poly"],
                        help="Activation applied to FFN after first projection")

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


def _register_layer_output_stat_hooks(model: nn.Module):
    output_stats = {}
    handles = []

    def _make_hook(name):
        def _hook(module, inputs, output):
            if output is None:
                return
            x = output[0] if isinstance(output, tuple) else output
            if x is None:
                return
            with torch.no_grad():
                x = x.detach().float()
                mean = x.mean()
                std = x.std(unbiased=False)

                if dist.is_available() and dist.is_initialized():
                    mean_all = mean.clone()
                    std_all = std.clone()
                    dist.all_reduce(mean_all, op=dist.ReduceOp.SUM)
                    dist.all_reduce(std_all, op=dist.ReduceOp.SUM)
                    mean = mean_all / dist.get_world_size()
                    std = std_all / dist.get_world_size()

                mean_val = mean.item()
                std_val = std.item()

                output_stats[name] = {
                    "mean": mean_val,
                    "std": std_val,
                }

                if module.training and hasattr(module, "x_std_ema"):
                    if module.x_std_ema is None:
                        module.x_std_ema = std_val
                    else:
                        module.x_std_ema = 0.95 * module.x_std_ema + 0.05 * std_val
        return _hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(_make_hook(f"layer{layer_idx}.x_l_plus_1")))

    return output_stats, handles


def _log_layer_output_stats(output_stats, update_step):
    logger.info(f"Layer output stats at update step {update_step}")
    for name in sorted(output_stats.keys()):
        stats = output_stats[name]
        logger.info(
            f"{name:20} x_(l+1) mean={stats['mean']:.6e}, x_(l+1) std={stats['std']:.6e}"
        )


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    was_training = model.training
    model.eval()
    try:
        _time = time.time()
        local_dir = "/home/qjw/code/python_code/LLM/mix_LN/MixLN-main/c4/allenai-c4"
        val_data = datasets.load_dataset(local_dir, split="validation", streaming=True, trust_remote_code=False)
        val_data = val_data.shuffle(seed=42)
        logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

        target_eval_tokens = 10_000_000
        evaluated_on_tokens = 0
        total_loss = torch.tensor(0.0, device=device)
        total_batches = 0
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

        for batch in val_data_mapped.batch(batch_size=batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            loss = model(**batch, labels=labels).loss
            total_loss += loss.detach()
            total_batches += 1

            local_batch_tokens = (batch["input_ids"] != pad_idx).sum().to(device=device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_batch_tokens, op=dist.ReduceOp.SUM)
            evaluated_on_tokens += local_batch_tokens.item()

            if evaluated_on_tokens >= target_eval_tokens:
                break

        total_loss = total_loss / max(1, total_batches)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss / world_size

        return total_loss.item(), evaluated_on_tokens
    finally:
        if was_training:
            model.train()



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    # keep Mix-LN wandb project
    if global_rank == 0:
        wandb.init(project="mixln", name=args.run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # use current-version local dataset loading
    local_dir = "/home/qjw/code/python_code/LLM/mix_LN/MixLN-main/c4/allenai-c4"
    data = datasets.load_dataset(local_dir, split="train", streaming=True)

    seed_for_shuffle = 32

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    model = initial_embedding(model, args)
    layer_output_stats, projection_stat_handles = _register_layer_output_stat_hooks(model)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")

        from safetensors.torch import load_file
        state_dict = load_file(f"{args.continue_from}/model.safetensors")
        model.load_state_dict(state_dict)

        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    # keep current-version checkpoint/save strategy
    if global_rank == 0:
        init_dir = f"{args.save_dir}/model_0"
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"Saving initial model to {init_dir}")
        model.save_pretrained(init_dir)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    if 'galore' in args.optimizer.lower():
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        param_groups = [
            {'params': regular_params},
            {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}
        ]

    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    layer_wise_flag = False
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if global_rank == 0:
            print("world_size:", dist.get_world_size())
            print("is_ddp:", isinstance(model, torch.nn.parallel.DistributedDataParallel))

    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        current_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for layer in current_model.model.layers:
            layer.current_step = update_step
            layer.total_steps = args.num_training_steps

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        if global_rank == 0 and update_step % 1000 == 0:
            _log_layer_output_stats(layer_output_stats, update_step)

        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            tokenizer.save_pretrained(current_model_directory)
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                }, step=global_step)
                print(f"final_eval_loss: {total_loss:.2e}, final_eval_tokens: {evaluated_on_tokens:.2e}")
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            pass

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
            }, step=global_step)
        update_time = time.time()

    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
        }, step=global_step)
        logger.info(f"Final eval loss: {total_loss}")
        print(f"final_eval_loss: {total_loss:.2e}, final_eval_tokens: {evaluated_on_tokens:.2e}")

    for handle in projection_stat_handles:
        handle.remove()

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
