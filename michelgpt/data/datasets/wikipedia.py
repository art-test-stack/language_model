from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer.models import HGFBPETokenizer as Tokenizer
from michelgpt.settings import *

import re
from datasets import load_dataset, DownloadConfig


class WikipediaDataset(Dataset):
    
    def __init__(self, tokenizer: Tokenizer = Tokenizer(), verbose: bool = True) -> None:
        super().__init__(name="wikipedia", multiplier=1.)
        self.train_part = 'pretraining'
        
        print('Downloading Wikipedia dataset...')
        
        wikipedia = load_dataset(
            path = 'wikipedia',
            name='20220301.en',
            download_config = DownloadConfig(max_retries = 10)
        )["train"]
        
        wikipedia = wikipedia.map(
            lambda doc: {'text': self._clean_wikipedia(doc['text'])},
            desc = 'Cleaning wikipedia',
            num_proc = NUM_THREADS
        )
        
        self.dataset = wikipedia.filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
        
        # self.sizes['train'] = 0
                
        # for doc in self.dataset:
        # self.sizes['train'] += len(str(doc['text']).strip())
        
        self.save(tokenizer, verbose)
        print(f'Wikipedia dataset downloaded: {len(self.dataset):,} documents | {self.sizes["train"]:,} characters')
        

    def _clean_wikipedia(self, text: str) -> str:
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' )', ')')
        text = text.replace('( ', '(')
        text = text.replace(' ]', ']')
        text = text.replace('[ ', '[')
        
        text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
        
        array = list(text)
        start = True
        
        for i in range(len(array)):
            if array[i] == '"':
                    array[i] = '«' if start else '»'
                    start = not start
            return ''.join(array)