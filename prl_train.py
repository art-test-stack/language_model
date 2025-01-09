from michelgpt.train.trainer_fsdp import FSDPTrainer

from michelgpt.data.tok import TikTokenizer as Tokenizer
from michelgpt.settings import *

if __name__ == "__main__":
    # To clear cache: rm -rf ~/.cache/huggingface/datasets/
    tokenizer = Tokenizer()
    print(tokenizer.special_tokens[CONTROL_TOKENS.start_of_text])
    trainer = FSDPTrainer()
    trainer.fit()