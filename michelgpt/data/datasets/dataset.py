from michelgpt.data.clean import *
from michelgpt.data.tokenizer import Tokenizer
from michelgpt.settings import *

import torch
from torch.utils.data.dataset import Dataset

from datasets import Dataset as Datasets

import numpy as np
import numpy.typing as npt

import pickle
from tqdm import tqdm
from typing import Tuple
from enum import Enum


class DatasetMode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TokenizedDataset(Dataset):
    def __init__(self,
        dataset: Datasets,
        mode: DatasetMode = DatasetMode.TRAIN,
        max_content: int = MAX_CONTEXT
    ) -> None:
        super().__init__()
        self.mode = mode
        self.max_content = max_content
        self._adapt_sequences_to_max_content(dataset)

    def _adapt_sequences_to_max_content(self, dataset) -> None:
        adapted_data = []
        self.meta_data = []
        for id, tkn_data in enumerate(dataset):
            self.meta_data.append({"raw_id": id, "id_range": [id, id + len(tkn_data) - self.max_content - 1]})

        for i in range(0, len(tkn_data) - self.max_content - 1):
            adapted_data.append(tkn_data[i: i + self.max_content + 1])

        self.data = torch.Tensor(adapted_data)

    def __getitem__(self, index: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.data == None, "No data"
        if index == None:
            index = np.random.randint(len(self.data))
        # text_encoded = torch.Tensor(Tokenizer().encode(self.data[index], verbose=False))
        
        x, y = self.data[:, :-1], self.data[:, 1:]
        return x, y

    def __len__(self) -> int:
        return len(self.data)
    

class Dataset():

    def __init__(
        self,
        name: str = "",
        # dataset: Datasets | None = None,
        multiplier: int | float = 4,
    ) -> None:
        # self.training_part = ''
        self.name = name
        
        self.multiplier = float(multiplier)
        self.train_part = ""
        self.sizes = { "train": 0, "val": 0, "test": 0}

        self.dataset: Datasets = None

        self.trainset: TokenizedDataset = None
        self.valset: TokenizedDataset = None
        self.testset: TokenizedDataset = None
        
    def get_document(self, index: int | None = None):
        assert self.dataset == None, "No data"
        if index == None:
            index = np.random.randint(len(self.dataset))
        
        return CONTROL_TOKENS.start_of_text + clean_string(self.dataset[index]["text"]) + CONTROL_TOKENS.end_of_text
    
    def document_to_tokens(self, document: dict[str, str], tokenizer: Tokenizer = Tokenizer()) -> dict[str, npt.NDArray[np.uint16] | int]:
        tokens = [tokenizer.to_index[CONTROL_TOKENS.start_of_text], *tokenizer.encode(document['text'], verbose=False), tokenizer.to_index[CONTROL_TOKENS.end_of_text]]
        return {'tokens': np.array(tokens, dtype = np.uint16), 'size': len(tokens)}
    
    def save(self, tokenizer: Tokenizer = Tokenizer()) -> None:
        split_dataset = self.dataset.train_test_split(test_size = PRETRAINING_VAL_RATIO, shuffle = True)
        split_dataset['val'] = split_dataset.pop('test')
        
        tokenized = split_dataset.map(
            lambda doc: self.document_to_tokens(doc, tokenizer),
            desc = f"{self.name} tokenization",
            num_proc = NUM_THREADS
        )
        
        for split, documents in tokenized.items():
            total = 0
            ids = []
            
        for doc in tqdm(documents, desc=f'Saving {self.name} {split}'):
            ids.append({
                'start': total,
                'size': doc['size']
            })
            total += doc['size']
            
        with open(DATA_FOLDER.joinpath(self.train_part).joinpath(self.name).joinpath(f'{split}_ids.pkl'), 'wb') as file:
            pickle.dump(ids, file)

        batch_size = 1_024

        while batch_size >= len(documents):
            batch_size //= 2
            
        self.sizes[split] = int(np.sum(len(documents), dtype = np.uint64))
        path = DATA_FOLDER.joinpath(self.train_part).joinpath(self.name).joinpath(f'{split}.bin')
        file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (self.sizes[split],))
        
        i = 0

        for batch_i in tqdm(range(batch_size), desc = f'Saving {self.name} {split}'):

            batch = documents.shard(num_shards=batch_size, index=batch_i, contiguous=True).with_format('numpy')
            file_batch = batch
            size_batch = len(file_batch["text"])
            file[i:i + size_batch] = size_batch
            i += len(file_batch)

        file.flush()
        
        self.trainset = TokenizedDataset(tokenized["train"], mode=DatasetMode.TRAIN)
        self.valset = TokenizedDataset(tokenized["val"], mode=DatasetMode.VAL)
        self.testset = TokenizedDataset(tokenized["test"], mode=DatasetMode.TEST) if "test" in tokenized.keys() else None
        
        with open(DATA_FOLDER.joinpath(self.train_part, self.name, f'metadata.pkl'), 'wb') as file:
            pickle.dump({
                'training_part': self.train_part,
                'name': self.name,
                'size': self.sizes,
                'multiplier': self.multiplier
            }, file)