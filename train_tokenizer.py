from datasets import load_dataset
from michelgpt.settings import *
from michelgpt.data.tokenizer_custom import Tokenizer


def main():
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en")

    wiki_set = wikipedia_dataset['train']
    wiki_set_30k = [ text for text in wiki_set[:1000]["text"]]

    tk = Tokenizer()

    tk.train(wiki_set_30k, vocab_size=VOCAB_SIZE)
    tk.register_special_tokens(CONTROL_TOKENS_LIST)
    tk.save()

    encoded_text = tk.encode("very simple test")
    print(f'Encoded text: {encoded_text}')

    decoded_text = tk.decode(encoded_text)
    print(f'Decoded text: {decoded_text}')


if __name__ == "__main__":
    print("Use custom Tokenization")
    main()