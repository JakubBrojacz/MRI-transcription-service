import logging
import itertools
import json

from tqdm import tqdm
from Bio import pairwise2

import config
import metrics


# Method described in paper - alignmnet of 5-grams into whole hypothesis


f_logger = logging.getLogger("Method_Logger")


def string_overlap(s1, s2):
    for id_start in range(len(s1)):
        done = True
        if len(s2) < len(s1)-id_start:
            continue
        for s2_id in range(len(s1)-id_start):
            if s1[id_start+s2_id] != s2[s2_id]:
                done = False
                break
        if done:
            return id_start
    return -1


def test_with_params(dna1, g2p, l1, track, param1, param2, model=None):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''

    # f_logger.info(f'param1: {param1}')
    # f_logger.info(f'param2: {param2}')

    kgram_score = {}

    for kgram in tqdm(l1):
        dna2 = ''.join(g2p.transliterate(kgram).lower().split())
        alignment = pairwise2.align.localxs(
            dna1, dna2, param1, param2, one_alignment_only=True)[0]
        score = alignment.score / len(dna2)

        kgram_score[kgram] = (alignment, score)

    tmp_l = list(l1)
    tmp_l.sort(key=lambda x: kgram_score[x][1], reverse=True)

    used_kgrams = [""]
    last_used = 1
    hypo = track.hypothesis_phon
    ids = [0 for sign in hypo]
    start, end = 0, 0
    for kgram in tqdm(tmp_l):
        # print(ids[start:end])
        alignment, score = kgram_score[kgram]
        start = alignment.start
        end = alignment.end - alignment.seqA.count('-')

        if any(ids[start:end]):
            if ids[start] and ids[end-1]:
                continue
            tmp_kgram = (' '.join([
                used_kgrams[kgram_id]
                for kgram_id, _ in itertools.groupby(ids[start:end])
            ])).split()
            kgram_splitted = kgram.split()[1:-1]
            if ids[start]:
                prev_id = ids[start]
                overlap_start = string_overlap(tmp_kgram, kgram_splitted)
                if overlap_start == -1:
                    continue
                for id_i in range(start, end):
                    if ids[id_i] != 0:
                        ids[id_i] = last_used + 1
                    else:
                        ids[id_i] = last_used
                used_kgrams.append(
                    ' '.join(kgram_splitted[len(tmp_kgram)-overlap_start:]))
                used_kgrams.append(
                    ' '.join(kgram_splitted[:len(tmp_kgram)-overlap_start]))
                used_kgrams[prev_id] = ' '.join(tmp_kgram[:overlap_start])
                last_used += 2
            elif ids[end-1]:
                prev_id = ids[end-1]
                overlap_start = string_overlap(kgram_splitted, tmp_kgram)
                if overlap_start == -1:
                    continue
                for id_i in range(start, end):
                    if ids[id_i] != 0:
                        ids[id_i] = last_used + 1
                    else:
                        ids[id_i] = last_used
                used_kgrams.append(' '.join(kgram_splitted[:overlap_start]))
                used_kgrams.append(' '.join(kgram_splitted[overlap_start:]))
                used_kgrams[prev_id] = ' '.join(
                    tmp_kgram[len(kgram_splitted)-overlap_start:])
                last_used += 2

            continue
        ids[start:end] = [last_used for _ in range(end-start)]
        used_kgrams.append(kgram)
        last_used += 1

    fixed = ' '.join(
        used_kgrams[kgram_id]
        for kgram_id, _ in itertools.groupby(ids)
    )
    return fixed

