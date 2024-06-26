
from michelgpt.data.datasets.dataset import Dataset
from michelgpt.settings import *

import re
import pickle 
import numpy as np
from tqdm import tqdm
# from pathlib import Path
from datasets import load_dataset, DownloadConfig, concatenate_datasets

class WikipediaDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'wikipedia'
		self.multiplier = 4.0

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
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

		self.save_raw()
		print(f'Wikipedia dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')


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
	

	def save_raw(self) -> None:
		split_dataset = self.dataset.train_test_split(test_size = PRETRAINING_VAL_RATIO, shuffle = True)
		split_dataset['val'] = split_dataset.pop('test')

		for split, documents in split_dataset.items():

			total = 0
			ids = []

			for doc in tqdm(documents, desc = f'Saving {self.name} {split} ids'):

				ids.append({
					'start': total,
					'size': len(doc)
				})

				total += len(doc)

			with open(DATA_FOLDER.joinpath(self.training_part).joinpath(self.name).joinpath(f'{split}_ids.pkl'), 'wb') as file:
				pickle.dump(ids, file)

			batch_size = 1_024

			while batch_size >= len(documents):
				batch_size //= 2

			self.size[split] = int(np.sum(len(documents), dtype = np.uint64))
			path = DATA_FOLDER.joinpath(self.training_part).joinpath(self.name).joinpath(f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (self.size[split],))
			i = 0

			for batch_i in tqdm(range(batch_size), desc = f'Saving {self.name} {split}'):

				batch = documents.shard(num_shards = batch_size, index = batch_i, contiguous = True).with_format('numpy')
				file_batch = batch
				size_batch = len(file_batch["text"])
				file[i:i + size_batch] = size_batch
				i += len(file_batch)

			file.flush()