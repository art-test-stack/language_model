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

CONTROL_TOKENS_LIST = list(CONTROL_TOKENS.__dict__.values())[1:-3]
FORCED_TOKENS = ["AI"]

DATA_FOLDER = Path("data")
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = Path("output")
MODEL_FOLDER = Path("")
VOCAB_SIZE = 32_000
VOCAB_FILE = DATA_FOLDER.joinpath("vocab.json")
MAX_TOKEN_LENGTH = 16

# TOKEN_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # GPT 4 SPLIT
TOKEN_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT 2 SPLIT

# ------------- DRIVE -------------

SAVE_ON_DRIVE = True
DRIVE_FILE = ""

# ------------- MODEL -------------

VOCAB_SIZE = 32_000
MAX_CONTEXT = 64

NUM_HEADS = 2
NUM_LAYERS = 2

DIM_MODEL = 128
DIM_FFN = 4 * DIM_MODEL

DIM_HEAD = DIM_MODEL // NUM_HEADS
# DIM_KEY = DIM_MODEL // NUM_HEADS
# DIM_VALUE = DIM_MODEL // NUM_HEADS

DROPOUT = .1

MASK_VALUE = -1e9
LINEAR_BIAS = False
FLASH_ATTENTION = False # TODO: Not implemented

# ------------- TRAIN -------------

BATCH_SIZE = 128
NUM_THREADS = 16
PRETRAINING_VAL_RATIO = 1e-3

MAX_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WARMUP_ITERS = 2_000

WEIGHT_DECAY = .1
DECAY_ITERS = 100_000

BETA_1 = .9
BETA_2 = .95

EPSILON = 1e-8

