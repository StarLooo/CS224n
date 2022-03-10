#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """ Highway network for ConvNN
        - Relu
        - Sigmoid
        - gating mechanism from LSTM
    """

    def __init__(self, embed_size):
        """ Init Higway network

            @param char_embed_size (int): Embedding size of word, in handout,
                                     it's e_{word} (dimensionality)
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(in_features=embed_size, out_features=embed_size)
        self.gate = nn.Linear(in_features=embed_size, out_features=embed_size)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        """
            Take mini-batch of sentence of ConvNN

            @param X_conv_out (Tensor): Tensor with shape (batch_size, char_embed_size)
            @return X_highway (Tensor): combinded output with shape (batch_size, char_embed_size)
        """
        X_projection = F.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = torch.mul(X_projection, X_gate) + torch.mul(
            X_conv_out, 1 - X_gate)
        return X_highway

### END YOUR CODE
