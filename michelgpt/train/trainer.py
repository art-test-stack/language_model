from michelgpt.data.datasets.dataset import Dataset
from michelgpt.data.tokenizer import Tokenizer

from michelgpt.settings import *

import torch
from torch import nn, optim

import numpy as np
from typing import List
import random


class Trainer:

    def __init__(self, model, tokenizer: Tokenizer, optimizer=None):
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.max_sequence_length = self.model.max_content
        self.softmax = nn.Softmax(dim=-1)
        self.loss_function = nn.CrossEntropyLoss()


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

    def train(self, train_set: Dataset, test_set: Dataset, epochs: int, batch_size: int):
        loss_per_epoch = []
        for epoch in range(epochs):
            losses = []

            batches = []
            for _, batch in enumerate(train_set):
                sequence_tensor = torch.tensor(data[i: i + batch_size], dtype=torch.long)

                mask = torch.ones_like(sequence_tensor)
                mask[sequence_tensor == self.tokenizer.character_to_token('<pad>')] = 0

                batches.append((sequence_tensor, mask))

            for batch in batches:
                self.model.train()

                input_tensor = torch.zeros((batch_size, self.model.max_sequence_length + 1), dtype=torch.long)
                mask = torch.zeros((batch_size, self.model.max_sequence_length + 1), dtype=torch.long)

                for i, input_entry in enumerate(batch[0]):
                    input_tensor[i] = input_entry

                for i, mask_entry in enumerate(batch[1]):
                    mask[i] = mask_entry

                model_output, target = self(x=input_tensor, mask=mask)

                loss = self.loss_function(model_output.transpose(1, 2), target)

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()

                self.optimizer.zero_grad()

                losses.append(loss.item())

            epoch_loss = np.average(losses)
            loss_per_epoch.append(epoch_loss)
            print('Epoch:', epoch, 'Loss:', epoch_loss)

        return loss_per_epoch