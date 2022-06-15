import logging
import itertools
import json
import re
import functools

from tqdm import tqdm
from Bio import pairwise2

import config
import metrics


# get next word based on probability after previous 4 words and alignment of previous and next word at the beginning of hypothesis


f_logger = logging.getLogger("Method_Logger")


def test_with_params(dna1, g2p, l1, track, param1, param2, model=None):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''

    f_logger.info(f'param1: {param1}')
    f_logger.info(f'param2: {param2}')

    fixed = ['']
    position = 0
    dna1 = ':'+dna1
    all_words = list(model.reverse_pronounciation_dictionary)
    all_words.remove('')

    with tqdm(total=len(dna1)) as pbar:
        while position < len(dna1):
            pbar.n = position
            pbar.refresh()
            candidates = model.predict(fixed[-5:])
            if candidates:
                score_for_non_candidates = max(candidates.values())/5
            else:
                score_for_non_candidates = 1
            word_scores = {
                word: max(candidates.get(word,0), score_for_non_candidates)
                for word in all_words
            }
            word_positions = {
                word: position+1
                for word in all_words
            }
            for word in all_words:
                alignemnt_goal = f'{fixed[-1]}:{word[:8]}'
                alignment = pairwise2.align.globalms(
                    (dna1[position:position+len(alignemnt_goal)-1])[::-1],
                    alignemnt_goal[::-1],
                    3,
                    -3,
                    -2, -0.3,
                    one_alignment_only=True
                )[0]
                sep_id = alignment.seqB[::-1].index(':')
                gap_before_sep = ((alignment.seqA[::-1])[:sep_id]).count('-')
                word_positions[word] = max(sep_id-gap_before_sep, 1)+position
                word_scores[word] *= alignment.score
            best_word = max(all_words, key=lambda word: word_scores[word])
            fixed.append(best_word)
            position = word_positions[best_word]

    fixed = ' '.join(
        model.reverse_pronounciation_dictionary[word]
        for word in fixed
    )
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
