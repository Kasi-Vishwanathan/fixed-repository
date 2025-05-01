# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch import Tensor


class Seq2Seq(nn.Module):
    """
    Build a Sequence-to-Sequence model.

    Parameters:
        * `encoder`: Encoder of seq2seq model. e.g. RoBERTa
        * `decoder`: Decoder of seq2seq model. e.g. Transformer
        * `config`: Configuration of encoder model
        * `beam_size`: Beam size for beam search (optional)
        * `max_length`: Max length of target for beam search (optional)
        * `sos_id`: Start of symbol ID in target for beam search (optional)
        * `eos_id`: End of symbol ID in target for beam search (optional)
    """

    def __init__(
        self,
        encoder,
        decoder,
        config,
        beam_size: int = None,
        max_length: int = None,
        sos_id: int = None,
        eos_id: int = None
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        # Initialize buffer with lower triangular matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_length, max_length)))
            if max_length is not None
            else torch.tril(torch.ones((1024, 1024)))  # Default size if not provided
        )

    # ... (rest of the class implementation)