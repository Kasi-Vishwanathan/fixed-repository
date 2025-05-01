# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()  # Modern super() syntax
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None, cls=False):
        # Validate input
        if code_inputs is not None:
            inputs = code_inputs
        elif nl_inputs is not None:
            inputs = nl_inputs
        else:
            raise ValueError("Either code_inputs or nl_inputs must be provided")

        # Common processing for both paths
        attention_mask = inputs.ne(1)
        hidden_states = self.encoder(inputs, attention_mask=attention_mask)[0]

        if cls:  # Use CLS token when requested
            pooled = hidden_states[:, 0]
        else:    # Mean pooling implementation
            masked_outputs = hidden_states * attention_mask.unsqueeze(-1)
            sum_masked = masked_outputs.sum(dim=1)
            divisor = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            pooled = sum_masked / divisor

        return torch.nn.functional.normalize(pooled, p=2, dim=1)