import re
from datasets import load_dataset, DownloadConfig, concatenate_datasets
from claudegpt.data.datasets.dataset import Dataset
from claudegpt.settings import *

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

		self.dataset = self.dataset # .filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

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