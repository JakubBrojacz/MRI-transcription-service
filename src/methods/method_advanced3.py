import logging
import itertools
import functools

from tqdm import tqdm
from Bio import pairwise2
import numpy as np


# dynamic programming , based on advanced1, I think the best one at the moment
# reduce word to 10 first signs

# SOMETHING IS BROKEN


f_logger = logging.getLogger("Method_Logger")


def get_last_word(words, transitions, word_id, column_id, num_of_words):
    fixed_ids = [word_id]

    for trans in transitions[1:column_id][::-1]:
        fixed_ids.append(trans[fixed_ids[-1]])

    # print(fixed_ids)
    last_words = [
        words[idx]
        for idx, _ in itertools.groupby(fixed_ids[::-1])
    ][-num_of_words:]
    return last_words


def test_with_params(dna1, g2p, l1, track, param1, param2, model):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''
    # dna1 = dna1[:40]

    # m1, dna1 = test_utils.prepare_test_data()
    # model.model = m1


    f_logger.info("start method")
    np.random.seed(42)

    words = list(model.model)
    word_to_id = {
        word: idx
        for idx, word in enumerate(words)
    }
    last_column = np.array([
        0
        if i > 0
        else 1
        for i, _ in enumerate(words)
    ])

    last_column_base = np.array([
        0
        if i > 0
        else 1
        for i, _ in enumerate(words)
    ])
    word_len = [
        0
        for _ in words
    ]

    transitions = [
        [
            -1
            for a0 in words
        ]
        for a1 in dna1
    ]

    dna1_positions = list(range(len(dna1)))
    for position in tqdm(dna1_positions):
        # highest probability of next column (+corresponding entry in transitions) based on ngram
        next_column = [
            0 for _ in words
        ]
        # values of alignment in next column
        next_column_base = [
            0 for _ in words
        ]

        for word_id, word in enumerate(words):  # iterating over current column

            # prediction = model.predict([word])
            prediction = model.predict(get_last_word(words, transitions, word_id, position, 5))
            pred_array = np.array([
                1 for _ in words
            ])
            for next_word in prediction:
                pred_array[word_to_id[next_word]] = prediction[next_word]
            pred_array = pred_array / np.sum(pred_array)
            
            if word_len[word_id] > 1:
                pred_array *= 0.2
                pred_array[word_id] = pred_array[word_id]+0.8
            if word_len[word_id] > 0:
                pred_array *= 0.4
                pred_array[word_id] = pred_array[word_id]+0.6

            for next_word_id, next_word in enumerate(words):
                word_prediction = pred_array[next_word_id] * last_column[word_id]
                if last_column[word_id] < 0:
                    word_prediction = 0
                next_word_id = word_to_id[next_word]
                if next_column[next_word_id] < word_prediction:
                    next_column[next_word_id] = word_prediction
                    transitions[position][next_word_id] = word_id

        for word_id, word in enumerate(words):  # iterating over next column
            if next_column[word_id] == 0:
                word_len[word_id] = 0
                continue
            phon = g2p.transliterate(word).lower()[:10]
            alignment_goal = dna1[position:position+10]
            # phon = word
            alignment = pairwise2.align.globalmc(
                alignment_goal[::-1],
                phon[::-1],
                6,
                -6,
                functools.partial(gap_function, w1=-0.5, w2=-0.01),
                functools.partial(
                    gap_function_no_start_penalty, w1=-2, w2=-0.3),
                one_alignment_only=True
            )[0]
            score = (alignment.score-0.7)/len(phon)
            if score < 0:
                score = 0

            # check if inside word
            if word_len[word_id] > 0 and last_column_base[word_id] > score:
                score = last_column_base[word_id]
            else:
                tmp = next(
                    i
                    for (i, e) in enumerate(alignment.seqB)
                    if e != "-"
                )
                word_len[word_id] = len(alignment_goal)-tmp

            if word_len[word_id] > 0:
                word_len[word_id] -= 1  # one character was just consumed

            next_column_base[word_id] = score

        # multiplying probability by alignment
        for i in range(len(words)):
            last_column[i] = next_column[i]*next_column_base[i]
            last_column_base[i] = next_column_base[i]
        last_column = last_column / np.sum(last_column)

        # debug messages
        if position % 40 == 0:
            max_prob = 0
            max_id = -1
            for idx in range(len(last_column)):
                if last_column[idx] > max_prob:
                    max_prob = last_column[idx]
                    max_id = idx
            fixed_ids = [max_id]

            for trans in transitions[1:position][::-1]:
                fixed_ids.append(trans[fixed_ids[-1]])

            # print(fixed_ids)
            fixed = ' '.join(
                words[idx]
                for idx, _ in itertools.groupby(fixed_ids[::-1])
            )
            print()
            print(position)
            print(fixed)
            print(get_last_word(words, transitions, max_id, position, 5))

    # print(transitions)
    max_prob = 0
    max_id = -1
    for idx in range(len(last_column)):
        if last_column[idx] > max_prob:
            max_prob = last_column[idx]
            max_id = idx
    fixed_ids = [max_id]

    for trans in transitions[1:][::-1]:
        fixed_ids.append(trans[fixed_ids[-1]])

    # print(fixed_ids)
    fixed = ' '.join(
        words[idx]
        for idx, _ in itertools.groupby(fixed_ids[::-1])
    )
    print(fixed)
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
