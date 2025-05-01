# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch import Tensor


class Seq2Seq(nn.Module):
    """
    Build Sequence-to-Sequence model.

    Args:
        encoder: Encoder of seq2seq model (e.g. RoBERTa)
        decoder: Decoder of seq2seq model (e.g. Transformer)
        config: Configuration of encoder model
        beam_size: Beam size for beam search (optional)
        max_length: Max length of target for beam search (optional)
        sos_id: Start symbol ID for beam search (optional)
        eos_id: End symbol ID for beam search (optional)
    """
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

        if max_length is not None:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones((max_length, max_length), dtype=torch.uint8))
            )