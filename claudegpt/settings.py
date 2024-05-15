from pathlib import Path
# ------------- DATA -------------

CONTROL_TOKENS = [
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

DATA_FOLDER = Path("data")
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = Path("output")

# ------------- TRAIN -------------

NUM_THREADS = 16
PRETRAINING_VAL_RATIO = 1e-3