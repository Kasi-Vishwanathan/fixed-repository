import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

def _truncate_seq_pair(tok_a, tok_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    total = len(tok_a) + len(tok_b)
    if total <= max_length:
        return
    excess = total - max_length
    if len(tok_a) > len(tok_b):
        new_len_a = max(0, len(tok_a) - excess)
        del tok_a[new_len_a:]
    else:
        new_len_b = max(0, len(tok_b) - excess)
        del tok_b[new_len_b:]

def _truncate_seq_pair_two_length(tok_a, tok_b, max_a, max_b):
    """Truncates a sequence pair to respect individual and total length limits."""
    # First truncate to individual max lengths
    while len(tok_a) > max_a:
        tok_a.pop()
    while len(tok_b) > max_b:
        tok_b.pop()
    # Then truncate for total length limit
    total = len(tok_a) + len(tok_b)
    max_total = max_a + max_b
    while total > max_total:
        if len(tok_a) > len(tok_b):
            tok_a.pop()
        else:
            tok_b.pop()
        total = len(tok_a) + len(tok_b)

@dataclass
class InputFeature:
    """A single training/test feature for model input."""
    input_ids: list
    attention_mask: list
    token_type_ids: list = None
    label: int = None