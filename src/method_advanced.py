import logging
import itertools
import json
import re
import functools

from tqdm import tqdm
from Bio import pairwise2

import config
import metrics


f_logger = logging.getLogger("Method_Logger")


def test_with_params(dna1, g2p, l1, track, param1, param2, model=None):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''

    f_logger.info(f'param1: {param1}')
    f_logger.info(f'param2: {param2}')

    kgram_score = {}

    for kgram in tqdm(l1):
        dna2 = ''.join(g2p.transliterate(kgram).lower().split())
        alignment = pairwise2.align.localxs(
            dna1, dna2, param1, param2, one_alignment_only=True)[0]
        score = alignment.score / len(dna2)

        kgram_score[kgram] = (alignment, score)

    tmp_l = list(l1)
    tmp_l.sort(key=lambda x: kgram_score[x][1], reverse=True)

    hypo = track.hypothesis_phon

    kgram = tmp_l[0]
    alignment, score = kgram_score[kgram]

    start = alignment.start
    end = alignment.end - alignment.seqA.count('-')
    fixed = kgram.split()
    hypo = dna1

    # forward
    with tqdm(total=len(hypo)) as pbar:
        pbar.update(end)
        while end < len(hypo):
            # print(fixed)
            candidates = model.predict(fixed[-5:])
            # print(candidates)
            for c in candidates:
                alignment = pairwise2.align.globalmc(
                    (hypo[end:])[::-1],
                    g2p.transliterate(c).lower()[::-1],
                    6/len(c),
                    -6/len(c),
                    functools.partial(gap_function, w1=-0.5, w2=-0.01),
                    functools.partial(
                        gap_function_no_start_penalty, w1=-2, w2=-0.3),
                    one_alignment_only=True
                )[0]
                # print(alignment)
                candidates[c] = alignment
            # print(candidates)
            if len(candidates) == 0:
                break
            candidate = max(list(candidates), key=lambda c: candidates[c].score)
            fixed.append(candidate)
            # print(candidates[candidate])
            step = len(candidates[candidate].seqB) - \
                re.search(r'[^-]', candidates[candidate].seqB).start()
            pbar.update(step)
            end += step

    # backward
    with tqdm(total=len(hypo)) as pbar:
        pbar.update(end)
        while end < len(hypo):
            # print(fixed)
            candidates = model.predict(fixed[-5:])
            # print(candidates)
            for c in candidates:
                alignment = pairwise2.align.globalmc(
                    (hypo[end:])[::-1],
                    g2p.transliterate(c).lower()[::-1],
                    6/len(c),
                    -6/len(c),
                    functools.partial(gap_function, w1=-0.5, w2=-0.01),
                    functools.partial(
                        gap_function_no_start_penalty, w1=-2, w2=-0.3),
                    one_alignment_only=True
                )[0]
                # print(alignment)
                candidates[c] = alignment
            # print(candidates)
            if len(candidates) == 0:
                break
            candidate = max(list(candidates), key=lambda c: candidates[c].score)
            fixed.append(candidate)
            # print(candidates[candidate])
            step = len(candidates[candidate].seqB) - \
                re.search(r'[^-]', candidates[candidate].seqB).start()
            pbar.update(step)
            end += step

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
