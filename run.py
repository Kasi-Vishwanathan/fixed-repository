# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. All rights reserved.
# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see licenses for specific language terms).

"""
Fine-tuning transformer models for language modeling on text (GPT, GPT-2, BERT, RoBERTa).
GPT-family models use causal language modeling (CLM) while BERT/RoBERTa use masked language modeling (MLM).
"""

from __future__ import absolute_import