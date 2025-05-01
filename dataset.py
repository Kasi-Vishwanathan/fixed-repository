import random
import torch
from torch.utils.data import Dataset
import os
import pickle
import logging
import json
from tqdm import tqdm


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum combined length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_pair_two_length(tokens_a, tokens_b, max_length_a, max_length_b):
    """Truncates a sequence pair in place to their respective maximum lengths."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length_a + max_length_b:
            break
        if len(tokens_b) > max_length_b:
            tokens_b.pop()
        else:  # len(tokens_a) > max_length_a
            tokens_a.pop()

class InputFeatures:
    """A single set of training/test features for a sequence pair."""