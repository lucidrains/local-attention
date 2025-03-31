"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py 

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
"""

import random
import tqdm
import gzip
import numpy as np
import os

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from local_attention import LocalTransformer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 5*4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 2048
SEQ_LEN = 2048
LOG_INTERVAL = 1
DEVICE = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
BACKEND = 'nccl' # DDP settings

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # initialize distributed training
    dist.init_process_group(backend=BACKEND)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    DEVICE = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(DEVICE)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, loading data etc.
    seed_offset = ddp_rank # each process gets a different seed

    assert  GRADIENT_ACCUMULATE_EVERY % ddp_world_size == 0
    GRADIENT_ACCUMULATE_EVERY //= ddp_world_size
 
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)

# instantiate GPT-like decoder model
model = LocalTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    causal = True,
    local_attn_window_size = 256,
    max_seq_len = SEQ_LEN,
    use_dynamic_pos_bias = True
).to(DEVICE)

# wrap model for distributed training
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # unwrap DDP container if needed

# prepare enwik8 data

if master_process:
    with gzip.open('./data/enwik8.gz') as file:
        print("[Process 0] Unzipping enwik8.gz...")
        X = np.frombuffer(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        trX = torch.from_numpy(np.copy(trX)).to(DEVICE)
        vaX = torch.from_numpy(np.copy(vaX)).to(DEVICE)
else:
    trX = torch.empty(int(90e6), dtype=torch.uint8).to(DEVICE)
    vaX = torch.empty(int(5e6), dtype=torch.uint8).to(DEVICE)

if ddp:
    # Synchronize all processes
    dist.barrier()
    dist.broadcast(trX, src=0)
    dist.broadcast(vaX, src=0)

data_train, data_val = trX, vaX
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)

if ddp:
    # create distributed data loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler))
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler))
else:
    # create standard data loaders for single GPU or CPU
    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))
    val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False))

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE)

# training

for iter_num in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for micro_step in range(GRADIENT_ACCUMULATE_EVERY):
        if ddp:
            model.require_backward_grad_sync = (micro_step == GRADIENT_ACCUMULATE_EVERY - 1)
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    if iter_num % LOG_INTERVAL == 0 and master_process:
        print(f"iter {iter_num}: loss {loss.item():.4f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if iter_num % VALIDATE_EVERY == 0 and master_process:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if iter_num % GENERATE_EVERY == 0 and master_process:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f"{prime} \n\n {'*'*100}")

        sample = raw_model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(f"{output_str} \n\n {'*'*100}")
