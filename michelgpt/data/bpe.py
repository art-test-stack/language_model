from michelgpt.settings import *

import re
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import List, Dict


class BPETokenizer:
    def __init__(self, control_tokens: Dict[str, str] = CONTROL_TOKENS_DICT, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab = {}
        self.vocab_size_tgt = vocab_size

        self.token_to_id = {}
        self.create_token_to_id(control_tokens.values())

        self.control_tokens = control_tokens
        self.merges = [ (tok, '</w>') for tok in control_tokens.values()]
        
    def preprocess_text(self, text):
        text = text.lower()
        words = re.findall(r'⮜[^⮜⮞]+⮞|\w+|\s|[^\w\s]', text)
        return words

    def preprocess_dataset(self, dataset):
        words = []
        for _, set in enumerate(tqdm(dataset, "Preprocess dataset")):
            words += self.preprocess_text(set['text'])

        return words

    def get_vocab(self, text):
        vocab = Counter([' '.join(list(word)) + ' </w>' if word not in self.control_tokens.values() else word for word in text])
        return vocab

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_vocabulary_size(self, vocab):
        unique_tokens = set()
        for word in vocab:
            unique_tokens.update(word.split())
        return len(unique_tokens)

    def create_token_to_id(self, vocab):
        for word in vocab:
            tokens = word.split()
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)

    def create_vocab(self, text):
        print("Start creating vocab...")
        vocab = self.get_vocab(text)
        # merges = []
        pbar = tqdm(total = self.vocab_size_tgt)
        
        while self.get_vocabulary_size(vocab) < self.vocab_size_tgt:
            voc_size = self.get_vocabulary_size(vocab)
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab = self.merge_vocab(best, vocab)

            pbar.update(self.get_vocabulary_size(vocab) - voc_size)
        self.create_token_to_id(vocab)
        return vocab, self.merges

    # Encoding text using BPE merges and token_to_id
    def encode(self, text):
        text = ' '.join(list(text)) + ' </w>'
        for (a, b) in self.merges:
            bigram = re.escape(' '.join((a, b)))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            text = p.sub(''.join((a, b)), text)
        tokens = text.split()
        return [self.token_to_id[token] if token in self.token_to_id.keys() else self.token_to_id[self.control_tokens['unknown_token']] for token in tokens]

    # Decoding BPE encoded ids back to text
    def decode(self, ids):
        id_to_token = {v: k for k, v in self.token_to_id.items()}
        tokens = [id_to_token[id] if id in id_to_token.keys() else id_to_token[self.token_to_id[self.control_tokens['unknown_token']]] for id in ids]
        merge_pairs = [(a+b, (a, b)) for a, b in self.merges]
        for merge, (a, b) in reversed(merge_pairs):
            tokens = ' '.join(tokens)
            tokens = tokens.replace(merge, f'{a} {b}')
            tokens = tokens.split()
        return ''.join(tokens).replace(' </w>', '')
