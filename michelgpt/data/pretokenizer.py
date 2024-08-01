from michelgpt.settings import *

from tokenizers import NormalizedString, PreTokenizedString

from typing import List
import regex as re


def regex_pattern(split_pattern: str = TOKEN_SPLIT_PATTERN, special_tokens: List[str] = CONTROL_TOKENS_LIST):
    if len(special_tokens) == 0:
        return re.compile(split_pattern)
    
    special_tokens_pattern = "| ?".join(map(re.escape, special_tokens))
    reg = split_pattern
    re_compiled = re.compile(rf"""({special_tokens_pattern})|({reg})""")

    return re_compiled

def split(
        text: str, 
        split_pattern: str = TOKEN_SPLIT_PATTERN,
        special_tokens: List[str] = CONTROL_TOKENS_LIST,
    ) -> List[str]:

    if text == '':
        return []
    
    re_compiled = regex_pattern(split_pattern, special_tokens)
    words = [ w[0] if w[0] else w[1] for w in re.findall(re_compiled, text) ]

    return words


class PreTokenizer():
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def split(self, i: int, normalized_text: NormalizedString) -> List[NormalizedString]: 
        if self.verbose:
            print("Starting pretokenization")

        words = split(str(normalized_text))

        words = [NormalizedString(w) for w in words]

        if self.verbose:
            print("Number of words:", '{:,.0f}'.format(len(words)))
            print("PreTokenization finished")

        return words
        

    def pre_tokenize(self, text: PreTokenizedString) -> None:
        
        text.split(self.split)