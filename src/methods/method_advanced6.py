import logging

from tqdm import tqdm
from Bio import pairwise2
import numpy as np


f_logger = logging.getLogger("Method_Logger")


def get_alignment_score_position(hypo_phon, last_position, last_fixed_phon, next_fixed_candidate_phon):
    alignment_goal = hypo_phon[last_position:last_position +
                               len(last_fixed_phon)+min(10, len(next_fixed_candidate_phon))]
    fixed_alignemnt_candidate = f'{last_fixed_phon}:{next_fixed_candidate_phon[:10]}'
    alignment_goal = alignment_goal
    fixed_alignemnt_candidate = fixed_alignemnt_candidate
    alignment = pairwise2.align.globalms(
        alignment_goal,
        fixed_alignemnt_candidate,
        6,
        -6,
        # -0.5, -0.01,
        -2, -0.3,
        one_alignment_only=True
    )[0]
    score = (alignment.score)/len(fixed_alignemnt_candidate)
    if score < 0:
        score = 0

    sign_pos_in_alignemnt = alignment.seqB.find(':')
    num_of_spaces_in_hyp_before_special = alignment.seqA[:sign_pos_in_alignemnt].count(
        '-')
    next_position = last_position + sign_pos_in_alignemnt - \
        num_of_spaces_in_hyp_before_special
    return score, next_position


def test_with_params(dna1, g2p, l1, track, param1, param2, model):
    # dna1 = dna1[:40]

    # m1, dna1 = test_utils.prepare_test_data2()
    # model.model = m1

    f_logger.info("start method")
    np.random.seed(42)

    words = list(model.reverse_pronounciation_dictionary)

    fixed = [
        "mr"
    ]
    position = 0

    with tqdm(total=len(dna1)) as pbar:
        while position < len(dna1):
            max_score = -1
            max_position = position+1
            max_word = ""

            probabilities = model.predict(fixed)
            probabilities_sum = sum((
                probabilities[word]
                for word in probabilities
            ))

            for word in words:
                score, new_position = get_alignment_score_position(
                    dna1,
                    position,
                    fixed[-1],
                    word
                )
                score = score*(0.1+probabilities.get(word, 0) /
                               probabilities_sum)
                if score > max_score:
                    max_score = score
                    max_position = new_position
                    max_word = word

            position = max_position
            fixed.append(max_word)

            pbar.n = position
            pbar.refresh()

    return " ".join((
        model.reverse_pronounciation_dictionary[word]
        for word in fixed
    ))


# x is gap position in seq, y is gap length
def gap_function_no_start_penalty(x, y, w1, w2):
    if x == 0:
        return 0.0
    return gap_function(x, y, w1, w2)


def gap_function(x, y, w1, w2):  # x is gap position in seq, y is gap length
    if y == 0:  # No gap
        return 0.0
    return w1 + (y-1)*w2
