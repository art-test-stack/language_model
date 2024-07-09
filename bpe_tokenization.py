from datasets import load_dataset
from michelgpt.settings import *
from michelgpt.data.bpe import BPETokenizer

def main():
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en")

    wiki_set = wikipedia_dataset['train']
    wiki_set_30k = [ {'id': id, 'text': text } for id, text in zip(wiki_set[:30_000]["id"], wiki_set[:10_000]["text"])]

    tokenizer = BPETokenizer(vocab_size=100)


    words = tokenizer.preprocess_dataset(wiki_set_30k)

    vocab, merges = tokenizer.create_vocab(words)

    # Encoding example
    encoded_text = [tokenizer.encode(word) for word in tokenizer.preprocess_text("simple example")]
    print(f'Encoded text: {encoded_text}')

    # Decoding example
    decoded_text = [tokenizer.decode(tokens) for tokens in encoded_text]
    print(f'Decoded text: {decoded_text}')

if __name__=="__main__":
    print("Use custom BPE Tokenization")
    main()