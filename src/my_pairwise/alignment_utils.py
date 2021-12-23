import functools
import warnings
from collections import namedtuple

from Bio import BiopythonWarning
from Bio import pairwise2
import numpy as np


def my_align(
    sequenceA,
    sequenceB,
    match_fn,
    gap_A_fn,
    gap_B_fn
):
    if not isinstance(sequenceA, list):
        sequenceA = str(sequenceA)
    if not isinstance(sequenceB, list):
        sequenceB = str(sequenceB)

    matrices = my_make_score_matrix_generic(
        sequenceA,
        sequenceB,
        match_fn,
        gap_A_fn,
        gap_B_fn,
    )

    score_matrix, trace_matrix, best_score = matrices
    score_matrix = np.array(score_matrix)
    trace_matrix = np.array(trace_matrix)
    # print(score_matrix)
    # print(score_matrix[:,-1])
    # print(trace_matrix)
    # max_ids = np.argmax(score_matrix, axis=1)
    # score_matrix = np.take_along_axis(
    #     score_matrix,
    #     np.expand_dims(max_ids, axis=1),
    #     axis=1)
    # score_matrix = score_matrix.squeeze(axis=1)
    # trace_matrix = np.take_along_axis(
    #     trace_matrix,
    #     np.expand_dims(max_ids, axis=1),
    #     axis=1)
    # trace_matrix = trace_matrix.squeeze(axis=1)

    score_matrix = score_matrix[:, -1]
    trace_matrix = trace_matrix[:, -1]

    return (score_matrix, trace_matrix)


_PRECISION = 1000


def rint(x, precision=_PRECISION):
    """Print number with declared precision."""
    return int(x * precision + 0.5)


def my_make_score_matrix_generic(
    sequenceA,
    sequenceB,
    match_fn,
    gap_A_fn,
    gap_B_fn
):
    """Generate a score and traceback matrix (PRIVATE).

    This implementation according to Needleman-Wunsch allows the usage of
    general gap functions and is rather slow. It is automatically called if
    you define your own gap functions. You can force the usage of this method
    with ``force_generic=True``.
    """
    local_max_score = 0
    # Create the score and traceback matrices. These should be in the
    # shape:
    # sequenceA (down) x sequenceB (across)
    lenA, lenB = len(sequenceA), len(sequenceB)
    score_matrix, trace_matrix = [], []
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        trace_matrix.append([None] * (lenB + 1))

    # Initialize first row and column with gap scores. This is like opening up
    # i gaps at the beginning of sequence A or B.
    for i in range(lenA + 1):
        score = gap_B_fn(0, i)
        score_matrix[i][0] = score
        trace_matrix[i][0] = 0
    for i in range(lenB + 1):
        score = gap_A_fn(0, i)
        score_matrix[0][i] = score
        trace_matrix[0][i] = 0

    # Fill in the score matrix.  Each position in the matrix
    # represents an alignment between a character from sequence A to
    # one in sequence B.  As I iterate through the matrix, find the
    # alignment by choose the best of:
    #    1) extending a previous alignment without gaps
    #    2) adding a gap in sequenceA
    #    3) adding a gap in sequenceB
    for row in range(1, lenA + 1):
        for col in range(1, lenB + 1):
            # First, calculate the score that would occur by extending
            # the alignment without gaps.
            # fmt: off
            nogap_score = (
                score_matrix[row - 1][col - 1]
                + match_fn(sequenceA[row - 1], sequenceB[col - 1])
            )

            # fmt: on
            # Try to find a better score by opening gaps in sequenceA.
            # Do this by checking alignments from each column in the row.
            # Each column represents a different character to align from,
            # and thus a different length gap.
            # Although the gap function does not distinguish between opening
            # and extending a gap, we distinguish them for the backtrace.
            row_open = score_matrix[row][col - 1] + gap_A_fn(row, 1)
            row_extend = max(
                score_matrix[row][x] + gap_A_fn(row, col - x) for x in range(col)
            )

            # Try to find a better score by opening gaps in sequenceB.
            col_open = score_matrix[row - 1][col] + gap_B_fn(col, 1)
            col_extend = max(
                score_matrix[x][col] + gap_B_fn(col, row - x) for x in range(row)
            )

            best_score = max(nogap_score, row_open,
                             row_extend, col_open, col_extend)
            local_max_score = max(local_max_score, best_score)

            score_matrix[row][col] = best_score

            # trace matrix describes minimal number of letters from seqA
            # that are used in best alignment
            a1 = 999
            a2 = 999
            a3 = 999
            a4 = 999
            a5 = 999
            if rint(nogap_score) == rint(best_score):
                a1 = trace_matrix[row-1][col-1] + 1
            if rint(row_open) == rint(best_score):
                a2 = trace_matrix[row][col-1]
            if rint(row_extend) == rint(best_score):
                a3 = trace_matrix[row][col-1]
            if rint(col_open) == rint(best_score):
                a4 = trace_matrix[row-1][col]
                if a4 > 0:
                    a4 += 1
            if rint(col_extend) == rint(best_score):
                a5 = trace_matrix[row-1][col]
                if a5 > 0:
                    a5 += 1
            trace_matrix[row][col] = min(a1, a2, a3, a4, a5)

    return score_matrix, trace_matrix, best_score


# x is gap position in seq, y is gap length
def gap_function_no_start_penalty(x, y, w1, w2):
    if x == 0:
        return 0
    return gap_function(x, y, w1, w2)


def gap_function(x, y, w1, w2):  # x is gap position in seq, y is gap length
    if y == 0:  # No gap
        return 0
    return w1 + (y-1)*w2


if __name__ == '__main__':
    import phonemic
    g2p = phonemic.G2P()
    alignment_goal = 'akotaibrakstes'
    # alignment_goal = 'acbca'
    phon = 'anamkot'
    # phon = 'c'
    score, trace = my_align(
        alignment_goal[::-1],
        phon[::-1],
        pairwise2.identity_match(1, -1),
        functools.partial(gap_function, w1=-1, w2=-1),
        functools.partial(
            gap_function_no_start_penalty, w1=-1, w2=-1)
    )
    # print(score)
    # print(trace)

    alignment = pairwise2.align.globalmc(
        alignment_goal[::-1],
        phon[::-1],
        1,
        -1,
        functools.partial(gap_function, w1=-1, w2=-1),
        functools.partial(
            gap_function_no_start_penalty, w1=-1, w2=-1),
        one_alignment_only=True
    )[0]
    print(pairwise2.format_alignment(*alignment))
