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
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 2048
SEQ_LEN = 2048
MIN_GPU_MEMORY = 20 * 1024**3  # 20GB in bytes
# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def check_gpu_memory(min_memory_required):
    """
    Check if the current GPU has enough free memory.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    free_memory = torch.cuda.mem_get_info(local_rank)[0]  # Free memory in bytes
    return free_memory >= min_memory_required

def setup_distributed():
    # initialize distributed training

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

     # Check GPU memory
    if not check_gpu_memory(MIN_GPU_MEMORY):
        print(f"GPU {local_rank} does not have enough memory. Skipping...")
        dist.destroy_process_group()
        exit(0)

setup_distributed()

# instantiate GPT-like decoder model

model = LocalTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    causal = True,
    local_attn_window_size = 256,
    max_seq_len = SEQ_LEN,
    use_dynamic_pos_bias = True
).cuda()

# wrap model for distributed training
model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

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

# create distributed data loaders
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler=train_sampler))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, sampler=val_sampler))

# optimizer

optim = Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)
