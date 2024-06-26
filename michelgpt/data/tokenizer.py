from michelgpt.settings import *
from michelgpt.data.clean import POSSIBLE_CHARS

import tokenizers as tk
from tokenizers import normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

import json
from tqdm import tqdm



class Tokenizer:

    def __init__(self) -> None:
        self.vocab = {}

        if VOCAB_FILE.exists():
            self.load_from_vocab()
        
        self._add_control_tokens_to_vocab()
        self._set_control_tokens()


    def get_vocab(self):
        for id, token in self.vocab.items():
            print(f"{id=}: {token=}")
        
        return self.vocab


    def _set_control_tokens(self):
        # self.__dict__.update(CONTROL_TOKENS)
        for attr, value in CONTROL_TOKENS_DICT.items():
            setattr(self, value, self.vocab[value])
            # setattr(self.vocab, value, self.vocab[value])


    def create(self, dataset):
        
        self._create_vocab(dataset)
        self.clean_vocab()
        self.save_vocab()


    def _create_vocab(self, dataset):
        
        print('Creating vocabulary...')

        tokenizer = tk.Tokenizer(BPE(unk_token= CONTROL_TOKENS.unknown ))

        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.post_processor = TemplateProcessing(
            single=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text}",
            pair=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text} $B:1 {CONTROL_TOKENS.end_of_text}:1",
            special_tokens=[
                (f"{CONTROL_TOKENS.start_of_text}", 1),
                (f"{CONTROL_TOKENS.end_of_text}", 2),
            ],
        )

        trainer = BpeTrainer(
			vocab_size = int(VOCAB_SIZE * 1.1),
			show_progress = True,
            special_tokens=CONTROL_TOKENS_LIST
        )

        def batch_iterator(dataset, batch_size=1000):
            for i in tqdm(range(0, len(dataset), batch_size)):
                yield dataset[i : i + batch_size]["text"]

        print("Start training tokenizer...")
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))

        self.vocab = tokenizer.get_vocab()

        self.clean_vocab()
        self.vocab = {v: i for i,v in dict(sorted({ i:v for v, i in self.vocab.items()}.items())).items()}
        self.save_vocab()

    def _add_control_tokens_to_vocab(self):
        for token in CONTROL_TOKENS_DICT.values():
            token_value = 0
            if token not in self.vocab.keys():
                while token_value in self.vocab.values():
                    token_value += 1
                self.vocab[token] = token_value
        
        return self.vocab

    def clean_vocab(self):
        def is_valid(word):
            if len(word) > MAX_TOKEN_LENGTH:
                return False
            
            if any(c not in POSSIBLE_CHARS for c in word):
                return False
            return True
            
        self.vocab = {k: v for k, v in self.vocab.items() if is_valid(k)}

    def sort_vocab(self):

        self.vocab = {v: i for i,v in dict(sorted({ i:v for v, i in self.vocab.items()}.items())).items()}
        self.save_vocab()

    def save_vocab(self):
        with open(VOCAB_FILE, 'w') as vf:
            json.dump(self.vocab, vf)


    def load_from_vocab(self):
        with open(VOCAB_FILE, 'rb') as vf:
            self.vocab = json.load(vf)

    def encode(self, text: str):
        text_encoded = [ self.vocab[token] if token in self.vocab.keys() else self.vocab[self.unknown_token] for token in text ]
        return text_encoded
    
    def decode(self, text):
        pass