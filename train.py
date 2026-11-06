import os
import pickle
from dataclasses import asdict
from contextlib import nullcontext

import csv
import random
import numpy as np
import torch
from model import GPT, GPTConfig
from devices import select_device
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

# ====================
# Config
# ====================

# device
device = select_device()
dtype = torch.float16 if device == 'cuda' else torch.float32
use_amp = device == 'cuda'

# seed
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# data 
dataset = 'tinyshakespeare'
batch_size = 32
block_size = 256
# model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False
# adam optimizer
learning_rate = 3e-4
weight_decay = 0.1
# scheduler
min_lr = 3e-5
warmup_iters = 100
max_iters = 3000
# evaluation
eval_interval = 200
eval_batches = 20


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset)
checkpoint_path = os.path.join(data_dir, 'model.pt')

# On CUDA, the returned context enables mixed precision
def autocast_context():
    if use_amp:
        return torch.amp.autocast(device_type=device, dtype=dtype)
    return nullcontext()


def get_batch(data):
    # data [N]
    idx = np.random.randint(0, len(data) - block_size, (batch_size,)) # [B,]
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in idx.tolist()
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in idx.tolist()
    ])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, datasets):
    out = {}
    was_training = model.training
    model.eval()
    try:
        for split, data in datasets.items():
            losses = torch.zeros(eval_batches, device=device)
            for k in range(eval_batches):
                X, Y = get_batch(data)
                with autocast_context():
                    _, loss = model(X, Y)
                losses[k] = loss.detach()
            out[split] = losses.mean().item()
    finally:
        model.train(was_training)
    return out


def main():
    if not 0 < warmup_iters < max_iters:
        raise ValueError('Expected 0 < warmup_iters < max_iters.')
    if eval_interval <= 0 or eval_batches <= 0:
        raise ValueError('eval_interval and eval_batches must be positive.')
    print(f'Using device: {device}')

    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f'Missing {meta_path}. Run data/tinyshakespeare/prepare.py first.'
        )
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if isinstance(meta, dict):
        meta_vocab_size = meta['vocab_size']
    else:
        meta_vocab_size, _, _ = meta
    print(f"Vocab size = {meta_vocab_size} found inside {meta_path}")

    datasets = {}
    for split in ('train', 'val'):
        path = os.path.join(data_dir, f'{split}.bin')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Missing {path}. Run data/tinyshakespeare/prepare.py first.'
            )
        data = np.memmap(path, dtype=np.uint16, mode='r')
        if len(data) <= block_size:
            raise ValueError(f'{split}.bin must contain more than {block_size} tokens.')
        datasets[split] = data

    # init the model
    config = GPTConfig(
        block_size =block_size,
        vocab_size = meta_vocab_size,
        n_layer = n_layer,
        n_head = n_head,
        n_embd = n_embd,
        dropout = dropout,
        bias = bias
    )
    model = GPT(config).to(device)
    print(f"Model initialized with {model.get_num_params():,} parameters.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # scheduler setup
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / (warmup_iters + 1),
        total_iters=warmup_iters
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_iters - warmup_iters,
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_iters]
    )

    best_val_loss = float('inf')
    history = []
    model.train()

    for iter_num in range(max_iters + 1):
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            losses = estimate_loss(model, datasets)
            history.append({
                'iter_num': iter_num,
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'lr': optimizer.param_groups[0]['lr']
            })
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model': model.state_dict(),
                    'model_args': asdict(config),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'seed': seed,
                }, checkpoint_path)
                print(f"New best model saved with val loss {best_val_loss:.4f}")

        if iter_num == max_iters:
            break

        X, Y = get_batch(datasets['train'])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(): # mixed precision if device='cuda'
            _, loss = model(X, Y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # in case scaler skips optimizer update
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()

    with open(os.path.join(data_dir, 'history.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iter_num', 'train_loss', 'val_loss', 'lr'])
        writer.writeheader()
        writer.writerows(history)

if __name__ == '__main__':
    main()
