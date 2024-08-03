from michelgpt.train.model import MichelTransformer
from michelgpt.train.optimizer import AdamW

from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer_custom import Tokenizer

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
        self.iter = 0
        self.tokens = 0
        self.model = model
        self.epochs = 0
        self.loss = float('inf')
        self.accuracy = .0
        self.val_loss = .0
        self.val_accuracy = .0
        self.best_val_loss = float('inf')

        if optimizer is None:
            self.optimizer = AdamW(model.parameters())
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.max_sequence_length = self.model.max_content
        self.softmax = nn.Softmax(dim=-1)
        self.loss_function = nn.CrossEntropyLoss()

        self.metrics = {
            "time": [],
            "iter": [],
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
        self.metrics["iter"].append(self.iter)
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
        self.iter = self.metrics["iter"][-1]
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


    def fit(self, train_set: Dataset, test_set: Dataset, batch_size: int):
        self.time = time.time()

        while True:
            losses = []

            batches = []
            for _, batch in enumerate(train_set, 0):
                sequence_tensor = torch.tensor(data[i: i + batch_size], dtype=torch.long)

                mask = torch.ones_like(sequence_tensor)
                mask[sequence_tensor == self.tokenizer.character_to_token('<pad>')] = 0

                batches.append((sequence_tensor, mask))


                input_tensor = torch.zeros((batch_size, self.model.max_content + 1), dtype=torch.long)
                mask = torch.zeros((batch_size, self.model.max_content + 1), dtype=torch.long)

                for i, input_entry in enumerate(batch[0]):
                    input_tensor[i] = input_entry

                for i, mask_entry in enumerate(batch[1]):
                    mask[i] = mask_entry

                model_output, target = self.forward(x=input_tensor, mask=mask)

                loss = self.loss_function(model_output.transpose(1, 2), target)

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.iter()

                self.optimizer.zero_grad()

                losses.append(loss.item())

            epoch_loss = np.average(losses)
            self.loss.append(epoch_loss)
            print('iter:', self.iter, 'Loss:', epoch_loss)