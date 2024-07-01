import torch
from pathlib import Path

# ----------- PROCESSOR -----------
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

# ------------- DATA -------------

class CONTROL_TOKENS:
    unknown = '⮜unknown⮞'
    padding = '⮜padding⮞'
    start_of_text = '⮜start-of-text⮞' 
    tab = '⮜tab⮞' 
    new_line = '⮜new-line⮞' 
    human = '⮜human⮞' 
    system = '⮜system⮞' 
    user = '⮜user⮞' 
    assistant = '⮜assistant⮞' 
    end_of_text = '⮜end-of-text⮞'


CONTROL_TOKENS_DICT = {
    "unknown_token": '⮜unknown⮞',
    "padding_token": '⮜padding⮞',
    "start_of_text_token": '⮜start-of-text⮞', 
    "tab_token": '⮜tab⮞', 
    "new_line_token": '⮜new-line⮞', 
    "human_token": '⮜human⮞', 
    "system_token": '⮜system⮞', 
    "user_token": '⮜user⮞', 
    "assistant_token": '⮜assistant⮞', 
    "end_of_text_token": '⮜end-of-text⮞'
}


CONTROL_TOKENS_LIST = [
    '⮜unknown⮞', 
    '⮜padding⮞', 
    '⮜start-of-text⮞', 
    '⮜tab⮞', 
    '⮜new-line⮞', 
    '⮜human⮞', 
    '⮜system⮞', 
    '⮜user⮞', 
    '⮜assistant⮞', 
    '⮜end-of-text⮞'
]
# list(CONTROL_TOKENS.__dict__.values())

FORCED_TOKENS = ["AI"]
# [
#     '⮜unknown⮞', 
#     '⮜padding⮞', 
#     '⮜start-of-text⮞', 
#     '⮜tab⮞', 
#     '⮜new-line⮞', 
#     '⮜human⮞', 
#     '⮜system⮞', 
#     '⮜user⮞', 
#     '⮜assistant⮞', 
#     '⮜end-of-text⮞'
# ]

DATA_FOLDER = Path("data")
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = Path("output")
VOCAB_SIZE = 32_000
VOCAB_FILE = DATA_FOLDER.joinpath("vocab.json")
MAX_TOKEN_LENGTH = 16

# ------------- MODEL -------------

VOCAB_SIZE = 32 # 32_000
MAX_CONTEXT = 64

NUM_HEADS = 2
NUM_LAYERS = 2

DIM_MODEL = 128
DIM_FFN = 4 * DIM_MODEL

DIM_KEY = DIM_MODEL // NUM_HEADS
DIM_VALUE = DIM_MODEL // NUM_HEADS

DROPOUT = .1

MASK_VALUE = -1e9
LINEAR_BIAS = False
FLASH_ATTENTION = False # TODO: Not implemented

# ------------- TRAIN -------------

BATCH_SIZE = 128
NUM_THREADS = 16
PRETRAINING_VAL_RATIO = 1e-3