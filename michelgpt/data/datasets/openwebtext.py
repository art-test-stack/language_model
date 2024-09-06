from michelgpt.data.datasets.dataset import Dataset
from michelgpt.settings import NUM_THREADS
from datasets import load_dataset


class OpenWebTextDataset(Dataset):
    def __init__(self, multiplier: int | float = 1.) -> None:
        super().__init__("openwebtext", multiplier)

        dataset = load_dataset("openwebtext", num_proc=NUM_THREADS)
        

