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

import logging
import random

import ngram_model
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage
import method_simple
import method_advanced
import method_advanced2
import method_advanced3
import method_advanced4
import utils


def prepare_test_data():
    model = {
        '<START>': {
            'NUM': 5,
            'a': {
                'NUM': 4
            },
            'b': {
                'NUM': 4
            },
            'c': {
                'NUM': 4
            },
        },
        'a': {
            'NUM': 4,
            'c': {
                'NUM': 4
            }
        },
        'b': {
            'NUM': 4,
            'c': {
                'NUM': 4
            }
        },
        'c': {
            'NUM': 5,
            'a': {
                'NUM': 4
            },
            'b': {
                'NUM': 4
            }
        },
    }
    dna1 = 'acbca'
    return model, dna1


def prepare_test_data2():
    model = {
        '<START>': {
            'NUM': 5,
            'ala': {
                'NUM': 4
            },
            'karo': {
                'NUM': 4
            },
        },
        'ala': {
            'NUM': 4,
            'ma': {
                'NUM': 4
            }
        },
        'karo': {
            'NUM': 4,
            'ma': {
                'NUM': 4
            }
        },
        'ma': {
            'NUM': 5,
            'kota': {
                'NUM': 4
            },
            'psa': {
                'NUM': 4
            }
        },
        'kota': {
            'NUM': 5,
        },
        'psa': {
            'NUM': 5,
        },
        'alupsafixolaks': {
            'NUM': 5,
        },
    }
    dna1 = 'ala ma psa'
    return model, dna1


def test_xd():
    ref = "mr mózgowia badanie wykonano w sekwencjach echa spinowego oraz sekwencji flair"

    g2p = phonemic.G2P()

    model = ngram_model.NGramModel(5)
    model.add_document(ref)

    fixed = method_advanced4.test_with_params(
        ref,
        g2p,
        None,
        None,
        None,
        None,
        model)

    print(ref)
    print(fixed)


if __name__ == '__main__':
    test_xd()