#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,
                 char_embed_size: int = 50,
                 m_word: int = 21,
                 k: int = 5,
                 f: int = None):
        """ 
        Init CNN which is a 1-D cnn.

        @param char_embed_size (int): embedding size of char (dimensionality), in other words
                                      the e_char in pdf
        @param k: kernel size, also called window size
        @param f: number of filters, in this assignment, it's the char_embed_size of word, in
                  other words the e_word in pdf
        """

        # Conv1d: https://pytorch.org/docs/stable/nn.html?highlight=conv1d#torch.nn.functional.conv1d
        # MaxPool1d
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=char_embed_size, out_channels=f, kernel_size=k)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        """
        map from X_reshaped to X_conv_out

        @param X_reshaped (Tensor): Tensor of char-level embedding with shape (batch_size, e_char, m_word),
                                    where e_char is char_embed_size of char,
                                    m_word is max_word_length.
        @return X_conv_out (Tensor): Tensor of word-level embedding with shape (batch_size, e_word)
        """

        X_conv = self.conv1d(X_reshaped)  # shape: (batch_size, e_word, m_word-k+1)
        X_conv = self.relu(X_conv)
        X_conv_out = self.maxpool(F.relu(X_conv))  # shape: (batch_size, e_word, 1)

        return torch.squeeze(X_conv_out, dim=-1)

### END YOUR CODE
