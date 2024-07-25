from michelgpt.settings import CONTROL_TOKENS_LIST

from tokenizers import NormalizedString
from typing import List
import regex


class PreTokenizer():
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def pre_tokenize(self, text: str) -> List[str]:
        if self.verbose:
            print("Starting pretokenization")

        if text == '':
            return []
        
        safe_control_tokens = [regex.escape(c) for c in CONTROL_TOKENS_LIST]
        reg = r'(' + r'|'.join(safe_control_tokens) + r'|\d+|\s+|\p{L}+|[^\d\p{L}\s' + r''.join([f'[{i}]' for i in safe_control_tokens]) + r']+)'
        
        text = text.lower()
        words = regex.split(reg, text, flags = regex.UNICODE, concurrent = False)

        values_to_remove = ['', ' ', '  ', None]
        
        words = [ 
            ' ' + w[:-1] if not w.startswith(' ') and w.endswith(' ') and w not in CONTROL_TOKENS_LIST else
            ' ' + w if not w.startswith(' ') and w not in CONTROL_TOKENS_LIST else
            w[:-1] if w.endswith(' ') and w not in CONTROL_TOKENS_LIST else w
            for w in words
        ]

        filter_by = lambda w: w not in values_to_remove
        words = list(filter(filter_by, words))

        words = [NormalizedString(w) for w in words]

        if self.verbose:
            print("Number of words:", '{:,.0f}'.format(len(words)))
            print("PreTokenization finished")

        return words