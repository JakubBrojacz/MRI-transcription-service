import logging
from turtle import position

from tqdm import tqdm
from Bio import pairwise2
import numpy as np
import functools


# recursive tree-based


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


class RecTesting:
    def __init__(self, model, dna1, max_depth):
        self.best_fixed = ""
        self.best_score = -1
        self.best_position = 0
        self.model = model
        self.dna1 = dna1
        self.max_depth = max_depth

    def finish(self, fixed):
        f_str = ''.join(fixed)
        alignment = pairwise2.align.globalms(
            self.dna1,
            f_str,
            6,
            -6,
            -2, -0.3,
            one_alignment_only=True
        )[0]
        score = (alignment.score)/len(f_str)
        if score > self.best_score:
            self.best_score = score
            self.best_fixed = " ".join(fixed)
            self.best_position = position
            # print(self.best_fixed)

    def finish_partial(self, fixed, position):
        f_str = ''.join(fixed)
        alignment = pairwise2.align.globalmc(
            self.dna1[::-1],
            f_str[::-1],
            6,
            -6,
            functools.partial(gap_function, w1=-0.5, w2=-0.01),
            functools.partial(
                gap_function_no_start_penalty, w1=-2, w2=-0.3),
            one_alignment_only=True
        )[0]
        score = (alignment.score)/len(f_str)
        if score > self.best_score:
            self.best_score = score
            self.best_fixed = " ".join(fixed)
            self.best_position = position
            # print(self.best_fixed)

    def test_recursive(self, fixed, position, depth=0):
        if position > (len(self.dna1)-5) or depth >= self.max_depth:
            self.finish_partial(fixed, position)
            return

        probabilities = self.model.predict(fixed)
        probabilities_sum = sum((
            probabilities[word]
            for word in probabilities
        ))

        candidates = (
            (
                word,
                get_alignment_score_position(
                    self.dna1,
                    position,
                    fixed[-1],
                    word
                )
            )
            for word in probabilities
        )

        candidates = [
            (
                word,
                ali[0] * (0.1+probabilities[word] /
                          probabilities_sum),
                ali[1]
            )
            for word, ali in candidates
        ]

        candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )[:5]
        # print(candidates)

        for id, (word, score, new_position) in enumerate(candidates):
            if id == 0 or score > 0.7:
                self.test_recursive(fixed + [word], new_position, depth+1)


def test_with_params(dna1, g2p, l1, track, param1, param2, model):
    # dna1 = dna1[:40]

    # m1, dna1 = test_utils.prepare_test_data2()
    # model.model = m1

    f_logger.info("start method")
    np.random.seed(42)

    words = list(model.reverse_pronounciation_dictionary)

    position = 0
    fixed = ["mr"]
    fixed_for_rt = ["mr"]
    with tqdm(total=len(dna1)) as pbar:
        while position < (len(dna1)-5):
            rt = RecTesting(model, dna1[position:], 5)
            rt.test_recursive(
                fixed_for_rt,
                0
            )
            rt_fixed = rt.best_fixed.split()
            _, new_position_offset = get_alignment_score_position(
                dna1[position:],
                0,
                rt_fixed[0],
                rt_fixed[1]
            )
            position += new_position_offset
            fixed.append(rt_fixed[1])
            fixed_for_rt = [rt_fixed[1]]
            # print(fixed)
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
