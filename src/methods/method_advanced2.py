import logging
import itertools
import json
import re
import time
import functools

from tqdm import tqdm
from Bio import pairwise2
import numpy as np
from hmmlearn import hmm

import config
import metrics


f_logger = logging.getLogger("Method_Logger")


def test_with_params(dna1, g2p, l1, track, param1, param2, model):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''
    dna1 = dna1[:40]

    f_logger.info("start method")
    np.random.seed(42)

    f_logger.info("setup hidden states")
    hidden_states = list(model.model)
    n_hidden_states = len(hidden_states)
    hidden_states_startprob = [
        0
        if i > 0
        else 1
        for i, _ in enumerate(hidden_states)
    ]
    
    f_logger.info("setup transmat")
    transmat = []
    for word in tqdm(hidden_states):
        prediction = model.predict([word])
        word_transmat = np.array([
            prediction[word_next]
            if word_next in prediction
            else 0
            for word_next in hidden_states
        ])
        if np.sum(word_transmat) < 1:
            word_transmat = word_transmat*0+1
        word_transmat = word_transmat / np.sum(word_transmat)
        transmat.append(word_transmat)

    f_logger.info("setup emissionprob")
    emissionprob = []
    for word in tqdm(hidden_states):
        word_emissions = []
        for position in range(len(dna1)):
            phon = g2p.transliterate(word).lower()
            alignment = pairwise2.align.globalmc(
                    (dna1[position:])[::-1],
                    phon[::-1],
                    6,
                    -6,
                    functools.partial(gap_function, w1=-0.5, w2=-0.01),
                    functools.partial(
                        gap_function_no_start_penalty, w1=-2, w2=-0.3),
                    one_alignment_only=True
                )[0]
            # print(pairwise2.format_alignment(*alignment))
            # input()
            score = (alignment.score-0.7)/len(phon)
            if score < 0:
                score = 0
            word_emissions.append(score)
        word_emissions = np.array(word_emissions)
        if np.sum(word_emissions) < 1:
            word_emissions = word_emissions*0+1
        word_emissions = word_emissions/np.sum(word_emissions)
        emissionprob.append(word_emissions)

    f_logger.info("setup model")
    HMMmodel = hmm.MultinomialHMM(n_components=n_hidden_states)
    HMMmodel.startprob_ = np.array(hidden_states_startprob)
    HMMmodel.transmat_ = np.array(transmat)
    HMMmodel.emissionprob_ = np.array(emissionprob)

    f_logger.info("predict")
    X = list(range(len(dna1)))
    X = np.array(X).reshape(1, -1)
    t1 = time.time()
    fixed_ids = HMMmodel.predict(X)
    t2 = time.time()
    f_logger.info(fixed_ids)
    fixed = ' '.join([
        hidden_states[idx] 
        for idx in fixed_ids
    ])
    f_logger.info(fixed)
    f_logger.info(f'function:HMMmodel.predict, took: {t2-t1} sec')
    input()

    fixed = ' '.join(fixed)
    return fixed


# x is gap position in seq, y is gap length
def gap_function_no_start_penalty(x, y, w1, w2):
    if x == 0:
        return 0
    return gap_function(x, y, w1, w2)


def gap_function(x, y, w1, w2):  # x is gap position in seq, y is gap length
    if y == 0:  # No gap
        return 0
    return w1 + (y-1)*w2
