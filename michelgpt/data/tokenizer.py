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

import json
import regex as re
from typing import Dict, List
from tqdm import tqdm


class Tokenizer:

    def __init__(
            self, 
            split_pattern: str = TOKEN_SPLIT_PATTERN, 
            file: Path = DATA_FOLDER,
            special_tokens: List[str] | str = CONTROL_TOKENS_LIST
        ) -> None:
        
        self.to_index: Dict[str, int] = {}
        self.to_token: Dict[int, str] = {}
        self.control_tokens: List[str]
        
        self.special_tokens = list(special_tokens) if type(special_tokens) == str else special_tokens

        self.file: Path = file
        if file.joinpath('vocab.json').exists():
            self.load_from_vocab(load_text_array(file.joinpath('vocab.txt')))
        else:
            self.create(file.joinpath("tk_data.txt"))
            save_text_array(self.vocab, file.joinpath('vocab.txt'))

        if file.joinpath('trainer.pt').exists():
            self.trainer: Trainer = None
        else:
            self.trainer: Trainer = None
            
        self.split_pattern = split_pattern
        self.compiled_pattern = re.compile(self.pattern)


    def get_vocab(self, type: str = "dict", verbose: bool = False):
        if verbose:
            for id, token in self.to_token.items():
                print(f"{id=}: {token=}")
        
        if type == "dict":
            return self.to_token
        
        elif type == "list":
            return self.to_token.values()
        
        elif type == "none":
            return


    def create(self, dataset):
        
        self._create_vocab(dataset)
        self.clean_vocab()
        self.save_vocab()


    def _create_vocab(self, dataset):
        
        print('Creating vocabulary...')

        tokenizer = tk.Tokenizer(BPE(unk_token=CONTROL_TOKENS.unknown))

        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        
        tokenizer.pre_tokenizer = PreTokenizer.custom(pretk.PreTokenizer())

        tokenizer.post_processor = TemplateProcessing(
            single=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text}",
            pair=f"{CONTROL_TOKENS.start_of_text} $A {CONTROL_TOKENS.end_of_text} $B:1 {CONTROL_TOKENS.end_of_text}:1",
            special_tokens=[
                (f"{CONTROL_TOKENS.start_of_text}", 1),
                (f"{CONTROL_TOKENS.end_of_text}", 2),
            ],
        )

        trainer = BpeTrainer(
			vocab_size = VOCAB_SIZE,
			show_progress = True,
            special_tokens=CONTROL_TOKENS_LIST
        )

        def batch_iterator(dataset, batch_size=1000):
            for i in tqdm(range(0, len(dataset), batch_size)):
                yield dataset[i : i + batch_size]["text"]

        print("Start training tokenizer...")
        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer, length=len(dataset))

        vocab = tokenizer.get_vocab()

        self.clean_vocab()
        self.vocab = {v: i for i,v in dict(sorted({ i:v for v, i in self.vocab.items()}.items())).items()}
        self.save_vocab()


    def _add_control_tokens_to_vocab(self):
        for token in CONTROL_TOKENS_LIST:
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
        with open(self.file.joinpath('vocab.json'), 'w') as vf:
            json.dump(self.vocab, vf)


    def load_from_vocab(self):
        with open(self.file.joinpath('vocab.json'), 'rb') as vf:
            self.to_token = json.load(vf)
            self.to_index = {v: k for k, v in self.to_token.items()}


    def encode(
            self, 
            text: str, 
            clean_text: bool = True,
            keep_control_tokens: bool = True,
            verbose: bool = True
        ):
        if verbose:
            print("Pretokenize...")

        if clean_text:
            text = clean_string(text=text, keep_control_tokens=keep_control_tokens)

        text_chunks = pretk.split(text)

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
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text_bytes = unclean_string(text_bytes, keep_control_tokens)
        text = text_bytes.decode("utf-8", errors="replace")

        return text