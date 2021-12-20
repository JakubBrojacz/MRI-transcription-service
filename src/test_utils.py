import argparse
import itertools
import pathlib
import unicodedata
import json
import random
import functools
import sys
import os
import re
import logging


import pandas as pd
from tqdm import tqdm
from Bio import pairwise2

import data_loading.sound_preprocessing as sound_preprocessing
import julius
import config
import ngram_model
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage
import metrics
import data_loading.train_test_data as train_test_data
import method_simple



# model.model = {
    #     '<START>': {
    #         'NUM': 5,
    #         'a': {
    #             'NUM': 4
    #         },
    #         'b': {
    #             'NUM': 4
    #         },
    #         'c': {
    #             'NUM': 4
    #         },
    #     },
    #     'a': {
    #         'NUM': 4,
    #         'c': {
    #             'NUM': 4
    #         }
    #     },
    #     'b': {
    #         'NUM': 4,
    #         'c': {
    #             'NUM': 4
    #         }
    #     },
    #     'c': {
    #         'NUM': 5,
    #         'a': {
    #             'NUM': 4
    #         },
    #         'b': {
    #             'NUM': 4
    #         }
    #     },
    # }
    # dna1 = 'acbca'