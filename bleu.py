#!/usr/bin/env python3

'''
This script was adapted from the original version by hieuhoang1972 which is part of MOSES. 
'''

# $Id: bleu.py 1307 2007-03-14 22:22:36Z hieuhoang1972 $

'''Provides:

cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into a form usable by score_cooked().
score_cooked(alltest, n=4): Score a list of cooked test sentences.

score_set(s, testid, refids, n=4): Interface with dataset.py; calculate BLEU score of testid against refids.

The reason for breaking the BLEU computation into three phases cook_refs(), cook_test(), and score_cooked() is to allow the caller to calculate BLEU scores for multiple test sets as efficiently as possible.
'''

import sys
import math
import re
import xml.sax.saxutils
import subprocess
import os

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0