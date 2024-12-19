from michelgpt.train.arg import ModelArgs
from michelgpt.train.model import MichelTransformer
from michelgpt.train.optimizer import AdamW

from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer.models import HGFBPETokenizer as Tokenizer
from michelgpt.utils import get_logger, rank_log, verify_min_gpu_count
from michelgpt.settings import *

import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

from typing import Callable

# def verify_min_gpu_count(min_gpus: int = 2) -> bool:
#     """ verification that we have at least 2 gpus to run dist examples """
#     has_cuda = torch.cuda.is_available()
#     gpu_count = torch.cuda.device_count()
#     return has_cuda and gpu_count >= min_gpus

# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

class ParallelTrainer:
    
    def __init__(
            model: MichelTransformer,
            tokenizer: Tokenizer = Tokenizer(), 
            optimizer: optim.Optimizer | Callable = None, 
            padding_token: int = 1,
            device: torch.device = DEVICE
        ):
        tp_size = 2
        logger = get_logger()

        # understand world topology
        _rank = int(os.environ["RANK"])
        _world_size = int(os.environ["WORLD_SIZE"])


        print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
        assert (
            _world_size % tp_size == 0
        ), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


        # create a sharding plan based on the given world_size.
        dp_size = _world_size // tp_size

        # Create a device mesh with 2 dimensions.
        # First dim is the data parallel dimension
        # Second dim is the tensor parallel dimension.
        device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

        # rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
        tp_mesh = device_mesh["tp"]
        dp_mesh = device_mesh["dp"]

        # For TP, input needs to be same across all TP ranks.
        # while for SP, input can be different across all ranks.
        # We will use dp_rank for setting the random seed
        # to mimic the behavior of the dataloader.
        dp_rank = dp_mesh.get_local_rank()

        # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
        simple_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

        model = MichelTransformer(simple_config).cuda()

        # init model weights
        model.init_weights()

        # parallelize the first embedding and the last linear out projection
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "embedding": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                # "norm": SequenceParallel(),
                "model_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Replicate()
                ),
            }
        )

        for layer_id, transformer_block in enumerate(model.decoder_stack.layers):
            layer_tp_plan = {
                # "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.w_q": ColwiseParallel(),
                "attention.w_k": ColwiseParallel(),
                "attention.w_v": ColwiseParallel(),
                "attention.w_o": RowwiseParallel(output_layouts=Shard(1)),
                # "dropout": SequenceParallel(),
                "layer_norm": SequenceParallel(),
                "ffn": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "ffn.w_1": ColwiseParallel(),
                "ffn.w_2": ColwiseParallel(),
                "ffn.layer_norm": RowwiseParallel(output_layouts=Shard(1)),
            }

            # Adjust attention module to use the local number of heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.d_heads = attn_layer.d_heads // tp_mesh.size()

            # Custom parallelization plan for the model
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )

        # Init FSDP using the dp device mesh
        sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

        rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

        # Create an optimizer for the parallelized and sharded model.
        lr = 3e-3
        rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
        optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

        # Training loop:
        # Perform a num of iterations of forward/backward
        # and optimizations for the sharded module.
        rank_log(_rank, logger, "\nStarting 2D training...")
        num_iterations = 10
        batch_size = 2

        for i in range(num_iterations):
            # seeding with dp_rank to ensure identical inputs for TP groups
            torch.manual_seed(i + dp_rank)
            inp = torch.randint(32000, (8, 256), device="cuda")

            output = sharded_model(inp)
            output.sum().backward()
            optimizer.step()
            rank_log(_rank, logger, f"2D iter {i} complete")

        rank_log(_rank, logger, "2D training successfully completed!")