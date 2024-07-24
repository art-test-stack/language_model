from michelgpt.train.model import MichelTransformer

from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer import Tokenizer

from michelgpt.settings import *

import torch
from torch import nn, optim

import numpy as np
import time, pickle
from typing import List
from pathlib import Path


class Trainer():

    def __init__(self, model: MichelTransformer, tokenizer: Tokenizer, optimizer=None):
        super().__init__()

        self.time = .0
        self.step = 0
        self.tokens = 0
        self.model = model
        self.epochs = 0
        self.loss = float('inf')
        self.accuracy = .0
        self.val_loss = .0
        self.val_accuracy = .0
        self.best_val_loss = float('inf')

        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.max_sequence_length = self.model.max_content
        self.softmax = nn.Softmax(dim=-1)
        self.loss_function = nn.CrossEntropyLoss()

        self.metrics = {
            "time": [],
            "step": [],
            "tokens": [],
            "epochs": [],
            "accuracy": [],
            "loss": [],
            "val_accuracy": [],
            "val_loss": [],
            "best_val_loss": []
        }

    def save_metrics(self) -> None:

        self.metrics['time'].append(time.time() - self.time)
        self.metrics["step"].append(self.step)
        self.metrics["tokens"].append(self.tokens)
        self.metrics["epochs"].append(self.epochs)
        self.metrics["accuracy"].append(self.accuracy)
        self.metrics["loss"].append(self.loss)
        self.metrics["val_accuracy"].append(self.val_accuracy)
        self.metrics["val_loss"].append(self.val_loss)
        self.metrics["best_val_loss"].append(self.best_val_loss)

        if not OUTPUT_FOLDER.exists():
            OUTPUT_FOLDER.mkdir()

        pickle.dump(self.metrics, open(OUTPUT_FOLDER.joinpath('metrics.pkl'), 'wb'))
        self.time = time.time()



    def load_metrics(self, path: Path) -> None:
        if not path.exists():
            return
        
        self.metrics_history = pickle.load(open(OUTPUT_FOLDER.joinpath('metrics.pkl'), 'rb'))
        self.step = self.metrics["step"][-1]
        self.metrics = self.metrics["tokens"][-1]
        self.epochs = self.metrics["epochs"][-1]
        self.accuracy = self.metrics["accuracy"][-1]
        self.loss = self.metrics["loss"][-1]
        self.val_accuracy = self.metrics["val_accuracy"][-1]
        self.val_loss = self.metrics["val_loss"][-1]
        self.best_val_loss = np.min(self.metrics["val_loss"])


    def save_model(self, path: Path) -> None:
        if not path.exists():
            path.mkdir()
        torch.save(self.model.state_dict(), path.joinpath("model.pt"))
        torch.save(self.optimizer.state_dict(), path.joinpath("optimizer.pt"))

        if SAVE_ON_DRIVE:
            pass

    def load_model(self, path: Path) -> None:
        if not path.exists():
            return
        
        self.model.load_state_dict(torch.load(path.joinpath("model.pt"), map_location=DEVICE))
        self.optimizer.load_state_dict(torch.load(path.joinpath("optimizer.pt"), map_location=DEVICE))


    def next_token_probabilities(self, x, mask, temperature=1.0):
        logits = self.model(x, mask)[:, -1]

        if temperature != 1.0:
            logits = logits / temperature

        probabilities = self.softmax(logits)
        return probabilities
    

    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inp, mask)
        return output, target
    

    def find_previous_session(self):
        pass

