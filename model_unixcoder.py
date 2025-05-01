# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        inputs = code_inputs if code_inputs is not None else nl_inputs
        if inputs is None:
            raise ValueError("At least one of code_inputs or nl_inputs must be provided")
        
        attention_mask = inputs.ne(1)
        hidden_states = self.encoder(inputs, attention_mask=attention_mask).last_hidden_state
        
        # Mean pooling with epsilon safeguard
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        return torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)