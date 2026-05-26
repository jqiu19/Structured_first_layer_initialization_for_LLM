import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import initial_scheme


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeiT on ImageNet-1K")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, choices=["deit_tiny_patch16_224", "deit_small_patch16_224"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--initialization", type=str, default=None)
    parser.add_argument("--sfli-std", type=float, default=1.0)
    parser.add_argument("--sfli-hd-coeffs", type=float, default=0.08)
    parser.add_argument("--sfli-bias-step", type=float, default=0.0008)
    parser.add_argument("--sfli-bias-noise-std", type=float, default=0.00002)
    parser.add_argument("--resume", type=str, default=None)
    return _merge_config(parser.parse_args())


def _merge_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key, value in cfg.items():
        arg_key = key.replace("-", "_")
        if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
            setattr(args, arg_key, value)
    if args.output_dir is None:
        args.output_dir = cfg.get("output_dir", f"experiment_vit/output/{args.model}")
    return args


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def is_main(rank):
    return rank == 0


def log(rank, msg):
    if is_main(rank):
        print(msg, flush=True)


def init_wandb(args, rank):
    if not is_main(rank):
        return None

    run_name = f"{args.model}_{args.initialization}_{args.nproc_per_node}_{args.seed}_preln"
    return wandb.init(
        project="mixln",
        name=run_name,
        config=vars(args),
        dir=args.output_dir,
    )


def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        transforms.RandomErasing(p=0.25, value="random"),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return train_transform, val_transform


def build_loaders(args, rank, world_size):
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected ImageNet folders at {train_dir} and {val_dir}. "
            "Download ImageNet-1K manually and arrange it as data_dir/train and data_dir/val."
        )
    train_transform, val_transform = build_transforms(args.img_size)
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform) #ImageFolder 的标签不是从文件名里读出来的，而是从“子文件夹名”推断出来的

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def cosine_lr(base_lr, min_lr, epoch, step, steps_per_epoch, warmup_epochs, epochs):
    progress_epoch = epoch + step / max(steps_per_epoch, 1)
    if progress_epoch < warmup_epochs:
        return base_lr * progress_epoch / max(warmup_epochs, 1e-8)
    progress = (progress_epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, mixup_fn, args, epoch, rank, wandb_run=None):
    model.train()
    total_loss = 0.0
    start = time.time()
    for step, (images, targets) in enumerate(loader):
        lr = cosine_lr(args.lr, args.min_lr, epoch, step, len(loader), args.warmup_epochs, args.epochs)
        set_lr(optimizer, lr)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp) and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item()

        if is_main(rank) and step % 50 == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f} lr={lr:.6e}", flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": step,
                        "train/global_step": epoch * len(loader) + step,
                    }
                )
    return total_loss / max(len(loader), 1), time.time() - start


@torch.no_grad()
def evaluate(model, loader, device, rank, world_size):
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss_sum += criterion(outputs, targets)
        correct += (outputs.argmax(dim=1) == targets).sum()
        total += targets.numel()
    if world_size > 1:
        dist.all_reduce(correct)
        dist.all_reduce(total)
        dist.all_reduce(loss_sum)
    return (loss_sum / total).item(), (correct / total * 100.0).item()


def save_checkpoint(args, model, optimizer, scaler, epoch, rank):
    if not is_main(rank):
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
    )


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    args.nproc_per_node = world_size
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    log(rank, f"Training {args.model} on ImageNet-1K for {args.epochs} epochs")
    log(rank, f"Initialization: {args.initialization}")
    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
    )
    model = initial_scheme.apply_initialization(model, args)
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    mixup_active = args.mixup > 0 or args.cutmix > 0
    mixup_fn = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        label_smoothing=args.label_smoothing,
        num_classes=args.num_classes,
    ) if mixup_active else None
    criterion = SoftTargetCrossEntropy() if mixup_active else LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    wandb_run = init_wandb(args, rank)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, elapsed = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, mixup_fn, args, epoch, rank, wandb_run=wandb_run)
        val_loss, val_acc1 = evaluate(model, val_loader, device, rank, world_size)
        log(rank, f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc1={val_acc1:.2f} time={elapsed:.1f}s")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/acc1": val_acc1,
                    "train/epoch_time": elapsed,
                }
            )
        save_checkpoint(args, model, optimizer, scaler, epoch, rank)

    if wandb_run is not None:
        wandb_run.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
