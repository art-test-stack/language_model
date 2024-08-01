# from torch.utils.data.dataset import Dataset

from michelgpt.data.clean import *
from michelgpt.data.tokenizer_custom import Tokenizer
from michelgpt.settings import *

import os, random, pickle
from tqdm import tqdm
from abc import ABC
import numpy as np
import numpy.typing as npt

class Dataset(ABC):

	def __init__(self) -> None:

		self.dataset = None
		self.training_part = ''
		self.name = ''
		self.size = {'train': 0, 'val': 0}
		self.multiplier = 1.0

