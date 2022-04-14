import enum
import logging
import itertools
import json
import re
import functools
from sklearn.utils import resample

from tqdm import tqdm
from Bio import pairwise2
import numpy as np

import config
import metrics


# dynamic programming - only arrows between endings of the words
# aligning '<last_word>:<next_word>'

TRESHOLD = 0.01
MIN_PROBABILITY_POINTS = 0.3
MAX_PROBABILITY_POINTS = 1.5
ALIGNMENT_MATCH = 1
ALIGNMENT_MISMATCH = -0.5
ALIGNMENT_BREAK = -0.5
ALIGNMENT_BREAK_CONTINUE = -0.3
NUMER_OF_HYPS_PER_COLUMN = 2

f_logger = logging.getLogger("Method_Logger")


def get_alignment(hyp, hyp_id, word, next_word):
    alignemnt_goal = f'{word}:{next_word[:8]}'
    hyp_to_alignment = (hyp[hyp_id:hyp_id+2*len(alignemnt_goal)-1])
    alignment_s = pairwise2.align.globalms(
        hyp_to_alignment[::-1],
        alignemnt_goal[::-1],
        ALIGNMENT_MATCH,
        ALIGNMENT_MISMATCH,
        ALIGNMENT_BREAK,
        ALIGNMENT_BREAK_CONTINUE,
        one_alignment_only=True
    )
    if not alignment_s:
        return -1, 0
    alignment = alignment_s[0]
    sep_id = alignment.seqB[::-1].index(':')
    gap_before_sep = ((alignment.seqA[::-1])[:sep_id]).count('-')
    next_word_start = max(sep_id-gap_before_sep, 1)+hyp_id
    next_word_score = alignment.score
    return next_word_start, next_word_score


def get_history(pointer_table_1, pointer_table_2, start_w, start_h,
                words, max_his=1000):
    result = [words[start_w]]
    his = 1
    while his < max_his:
        his += 1
        start_w, start_h = pointer_table_1[start_w, start_h], \
            pointer_table_2[start_w, start_h]
        if start_w < 0:
            break
        result.append(words[start_w])
    return result[::-1]


def history_to_str(history, model):
    return " ".join([
        model.reverse_pronounciation_dictionary[word]
        for word in history
    ])


def get_model_probabilities(history, model, words):
    candidates = model.predict(history)
    if not candidates:
        return {
            word: MIN_PROBABILITY_POINTS
            for word in words
        }
    max_score = max(candidates.values())
    min_score = min(min(candidates.values()), 0.0)
    # Normalize values to 0.8-1
    word_scores = {
        word: (MAX_PROBABILITY_POINTS - MIN_PROBABILITY_POINTS) *
        (candidates.get(word, 0) - min_score) /
        ((max_score-min_score)) +
        MIN_PROBABILITY_POINTS
        for word in words
    }
    return word_scores


def test_with_params(hyp, g2p, l1, track, param1, param2, model=None):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''

    words = list(model.reverse_pronounciation_dictionary)

    dynamic_table = np.zeros((len(words), len(hyp)), dtype=float)
    pointer_table_1 = np.zeros((len(words), len(hyp)), dtype=int)
    pointer_table_2 = np.zeros((len(words), len(hyp)), dtype=int)
    start_pos_table = np.zeros((len(words), len(hyp)), dtype=int)
    dynamic_table[1, 0] = 1
    pointer_table_1[1, 0] = -1
    pointer_table_2[1, 0] = -1

    # go by columns
    for hyp_id in tqdm(range(len(hyp))):
        # first get only 4 best words at every position
        ids = np.argpartition(
            dynamic_table[:, hyp_id], -NUMER_OF_HYPS_PER_COLUMN)[-NUMER_OF_HYPS_PER_COLUMN:]
        for word_id in range(len(words)):
            if word_id not in ids:
                dynamic_table[word_id, hyp_id] = 0
        # for every available word find all next candidates
        for word_id, word in enumerate(words):
            if dynamic_table[word_id, hyp_id] < TRESHOLD:
                continue
            history = get_history(
                pointer_table_1, pointer_table_2,
                word_id, hyp_id,
                words,
                max_his=5)
            print(history_to_str(history, model))
            model_probs = get_model_probabilities(history, model, words)

            # for each candidate
            for next_word_id, next_word in enumerate(words):
                next_word_start, next_word_score = \
                    get_alignment(
                        hyp, start_pos_table[word_id, hyp_id], word, next_word)
                next_word_score *= model_probs[next_word]
                next_word_end = max(next_word_start+len(next_word), hyp_id+1)

                if next_word_score < TRESHOLD:
                    continue
                if next_word_end >= len(hyp):
                    continue

                next_word_score += dynamic_table[word_id, hyp_id]

                # if candidate has high enough score
                # and its higher than previous score in that place
                # place him in table
                if next_word_score < dynamic_table[next_word_id, next_word_end]:
                    continue
                dynamic_table[next_word_id, next_word_end] = next_word_score
                pointer_table_1[next_word_id, next_word_end] = word_id
                pointer_table_2[next_word_id, next_word_end] = hyp_id
                start_pos_table[next_word_id, next_word_end] = next_word_start

    # get max at the end
    word_id, hyp_id = np.unravel_index(
        dynamic_table.argmax(), dynamic_table.shape)

    history = get_history(pointer_table_1, pointer_table_2,
                          word_id, hyp_id, words, max_his=1000)
    return history_to_str(history, model)


# x is gap position in seq, y is gap length
def gap_function_no_start_penalty(x, y, w1, w2):
    if x == 0:
        return 0
    return gap_function(x, y, w1, w2)


def gap_function(x, y, w1, w2):  # x is gap position in seq, y is gap length
    if y == 0:  # No gap
        return 0
    return w1 + (y-1)*w2
