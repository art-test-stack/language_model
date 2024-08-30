from michelgpt.data.clean import *
import michelgpt.data.pretokenizer as pretk
from michelgpt.data.utils import *
from michelgpt.settings import *

import tokenizers as tk
from tokenizers import normalizers
from tokenizers.models import BPE
from tokenizers.trainers import Trainer, BpeTrainer

from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

import json, pickle
import regex as re
from typing import Dict, List
from tqdm import tqdm
from copy import copy

class Tokenizer:

    def __init__(
            self, 
            split_pattern: str = TOKEN_SPLIT_PATTERN, 
            file: Path = DATA_FOLDER,
            special_tokens: List[str] | str = CONTROL_TOKENS_LIST
        ) -> None:
        
        self.to_index: Dict[str, int] = {}
        self.to_token: Dict[int, str] = {}
        self.control_tokens: List[str] = special_tokens
        
        self.special_tokens: Dict[int, str] = {}

        self.file: Path = file
        if file.joinpath('vocab.json').exists():
            self.load_vocab()
        # else:
        #     self.create()
        #     save_text_array(self.vocab, file.joinpath('vocab.txt'))

        if file.joinpath('trainer.pt').exists():
            self.load_trainer()
        else:
            self.trainer: Trainer = BpeTrainer(
                vocab_size = VOCAB_SIZE,
                show_progress=True,
                special_tokens=CONTROL_TOKENS_LIST
            )
            self.save_trainer()
        self._add_special_tokens_to_vocab(special_tokens=special_tokens if type(special_tokens)==list else [special_tokens])
        self.split_pattern = split_pattern
        self.compiled_pattern = pretk.regex_pattern(self.split_pattern, special_tokens=self.control_tokens)


    def get_vocab(self, type: str = "dict", verbose: bool = False) -> Dict[int, str] | List[str] | None:
        if verbose:
            for id, token in self.to_token.items():
                print(f"{id=}: {token=}")
        
        if type == "dict":
            return self.to_token
        
        elif type == "list":
            return self.to_token.values()
        
        elif type == "none":
            return


    def create(self, dataset, batch_size: int = 2048, verbose: bool = True) -> None:
        
        self._create_vocab(dataset, batch_size, verbose)
        self.clean_vocab()
        self._add_special_tokens_to_vocab(copy(list(self.special_tokens.values())))
        self.sort_vocab()
        self.save_vocab()


    def _create_vocab(self, dataset, batch_size: int = 2048, verbose: bool = True):
        
        if verbose:
            print('Creating vocabulary...')

        tokenizer = tk.Tokenizer(BPE(unk_token=CONTROL_TOKENS.unknown))

        tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        
        tokenizer.pre_tokenizer = PreTokenizer.custom(pretk.PreTokenizer(verbose=False))

        tokenizer.post_processor = TemplateProcessing(
            single=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text}",
            pair=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text} $B:1 {CONTROL_TOKENS.end_of_text}:1",
            special_tokens=[
                (f"{CONTROL_TOKENS.start_of_text}", 1),
                (f"{CONTROL_TOKENS.end_of_text}", 2),
            ],
        )

        def batch_iterator(dataset, batch_size=batch_size):
            for i in tqdm(range(0, len(dataset), batch_size)):
                try: 
                    yield dataset[i : i + batch_size]["text"]
                except:
                    yield dataset["text"][i : i + batch_size]

        if verbose:
            print("Start training tokenizer...")
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=self.trainer, length=len(dataset))

        self.to_index = tokenizer.get_vocab()
        self.to_token = { idx: tok for tok, idx in self.to_index.items() }

        self.save_trainer()


    def _add_special_tokens_to_vocab(self, special_tokens: List[str] = CONTROL_TOKENS_LIST) -> None:
        to_pop = []
        for idx, token in self.special_tokens.items():
            if token not in self.to_token.values():
                to_pop.append(idx)
        
        [self.special_tokens.pop(idx) for idx in to_pop]

        new_tokens = []
        new_idx = []
        for token in special_tokens:
            token_idx: int = 0
            if token not in self.to_token.values():
                while token_idx in self.to_token.keys():
                    token_idx += 1
                new_idx.append(token_idx)
                new_tokens.append(token)
                self.to_token[token_idx] = token
                self.special_tokens[token_idx] = token
            elif token not in self.special_tokens.values():
                self.special_tokens[self.to_index[token]] = token
            # elif not self.special_tokens[self.to_index[token]] == token:
            #     pass

        self.to_index = { tok: idx for idx, tok in self.to_token.items() }


    def clean_vocab(self):
        def is_valid(word):
            return True
            # if len(word) > MAX_TOKEN_LENGTH:
            #     return False
            
            # if any(c not in POSSIBLE_CHARS for c in word):
            #     return False
            # return True
            
        self.to_token = { idx: tok for idx, tok in self.to_token.items() if is_valid(tok) }
        self.to_index = { tok: idx for idx, tok in self.to_token.items() }


    def sort_vocab(self):
        self.to_token = dict(sorted({ int(idx): tok for tok, idx in self.to_index.items()}.items()))
        self.to_index = { t: i for i, t in self.to_token.items() }
        self.save_vocab()


    def save_vocab(self):
        with open(self.file.joinpath('vocab.json'), 'w') as vf:
            json.dump(self.to_token, vf)


    def load_vocab(self):
        with open(self.file.joinpath('vocab.json'), 'rb') as vf:
            self.to_token = json.load(vf)
            self.to_index = {v: k for k, v in self.to_token.items()}


    def save_trainer(self):
        with open(self.file.joinpath('trainer.pkl'), 'wb') as file:
            pickle.dump(self.trainer, file)


    def load_trainer(self):
        self.trainer = pickle.load(self.file.joinpath('trainer.pkl'), 'rb')


    def encode(
            self, 
            text: str, 
            clean_text: bool = False,
            keep_control_tokens: bool = True,
            verbose: bool = True
        ):
        if verbose:
            print("Pretokenize...")

        if clean_text:
            text = clean_string(text=text, keep_control_tokens=keep_control_tokens)

        text_chunks = pretk.split(text, split_pattern=self.split_pattern, special_tokens=self.special_tokens.values())
        if verbose:
            print("Encoding dataset...")

        output = []

        for token in tqdm(text_chunks, disable=not verbose):
            if token in self.to_index.keys():
                output.append(self.to_index[token])
                continue
            j = 0
            while j < len(token):
                found = False
                for k in reversed(range(min(MAX_TOKEN_LENGTH, len(token) - j))):

                    word = token[j:j + k + 1]

                    if word in self.to_index:
                        output.append(self.to_index[word])
                        j += k
                        found = True
                        break

                if not found:
                    output.append(self.to_index[CONTROL_TOKENS.unknown])
                j += 1

        return output

    def decode(
            self, 
            token_ids: List[int] | torch.Tensor | int, 
            keep_control_tokens: bool = False
        ) -> str | List[str]:

        if type(token_ids) == int:
            token_ids = [token_ids]
        if type(token_ids) == torch.Tensor:
            token_ids = token_ids.detach().cpu().tolist()
        elif type(token_ids) != list:
            token_ids = list(token_ids)

        part_bytes = []

        for idx in token_ids:
            if idx in self.to_token.keys():
                part_bytes.append(self.to_token[idx])
            elif idx in self.special_tokens.keys():
                part_bytes.append(self.special_tokens[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
            
        text = "".join(part_bytes)
        text = unclean_string(text, keep_control_tokens) # .encode("utf-8")
        # text = text_bytes.decode("utf-8", errors="replace")

        return text