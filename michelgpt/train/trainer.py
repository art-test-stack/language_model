from michelgpt.train.model import MichelTransformer
from michelgpt.train.optimizer import AdamW

from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer_custom import Tokenizer

from michelgpt.settings import *

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import numpy as np
import time, pickle, wandb
from typing import List
from pathlib import Path


class Trainer():

    def __init__(
            self, 
            model: MichelTransformer, 
            tokenizer: Tokenizer = Tokenizer(), 
            optimizer=None, 
            padding_token=1,
            device: torch.device = DEVICE
        ):
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
        self.loss_function = nn.CrossEntropyLoss(ignore_index=padding_token)

        self.device = device
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

        if SAVE_ON_WANDB:
            wandb.init(
                project="michel-gpt-training",
                config={
                    "learning_rate": self.optimizer.learning_rate,
                    "architecture": "Transformer",
                    "dataset": "many",
                    "epochs": WARMUP_ITERS
                }
            )

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
    

    # def forward(self, x, mask):
    #     """
    #     Autoregressive forward pass
    #     """
    #     inp, target = x[:, :-1], x[:, 1:]
    #     mask = mask[:, :-1]

    #     output = self.model(inp, mask)
    #     return output, target
    

    def find_previous_session(self):
        pass


    def fit(self, dataset: Dataset, batch_size: int):
        self.time = time.time()

        train_set = DataLoader(
            dataset=dataset.trainset,
            batch_size=batch_size,
            shuffle=True
        )
        val_set = DataLoader(
            dataset.valset,
            batch_size=batch_size,
            shuffle=True
        )
        test_set = DataLoader(
            dataset.testset,
            batch_size=batch_size,
            shuffle=True
        ) if dataset.testset else val_set

        while True:
            losses = []

            for _, batch in enumerate(train_set, 0):
                
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                mask = torch.ones_like(batch).to(DEVICE)

                pred = self.model(x=x, mask=mask)
                loss = self.loss_function(y, pred)

                loss.backward()


            self.model.clean_nan()

            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

            epoch_loss = np.average(losses)
            self.losses.append(epoch_loss)

            if self.iter % VALIDATION_STEP == 0:
                pass