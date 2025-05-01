# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2-0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the time.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

from fractions import Fraction
import math
import numpy as np  # Added missing numpy import

def bleu_stats(candidate, reference):
    """Compute statistics for BLEU calculation."""
    stats = []
    stats.append(len(candidate))
    stats.append(len(reference))
    
    for n in range(1, 5):
        candidate_ngrams = get_ngrams(candidate, n)
        reference_ngrams = get_ngrams(reference, n)
        
        # Count matches
        matches = 0
        for ngram in candidate_ngrams:
            if ngram in reference_ngrams:
                matches += 1
        stats.append(matches)
        
        # Count possible
        possible = max(len(candidate) - n + 1, 0)
        stats.append(possible)
        
    return stats

def bleu(stats):
    """Compute BLEU given statistics."""
    if all(x == 0 for x in stats[2::2]):
        return 0.0  # Avoid division by zero in log_bleu_prec
    
    c, r = stats[:2]
    log_bleu_prec = sum(
        math.log(Fraction(num) / denom) 
        for num, denom in zip(stats[2::2], stats[3::2])
    ) / 4.0
    
    bypass_div = c / r  # Renamed to snake_case
    bp = math.exp(1 - bypass_div) if bypass_div < 1 else 1.0
    
    return bp * math.exp(log_bleu_prec)

def smoothing(stats):
    """Apply smoothing to statistics for smooth BLEU calculation."""
    smoothed = list(stats)
    for i in range(2, 10, 2):
        if smoothed[i] == 0:
            smoothed[i] += 1
            smoothed[i+1] += 1
    return smoothed

def get_ngrams(sequence, n):
    """Extract n-grams from a sequence."""
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]