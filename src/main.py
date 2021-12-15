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
import ASR_main_flow

f_logger = logging.getf_logger("File_Logger")
f_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
f_logger.addHandler(fh)

c_logger = logging.getf_logger("Console_Logger")
c_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
c_logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--input', '-I',
                        default=None,
                        help='Input audio file. Format: m4a',
                        type=pathlib.Path)
    parser.add_argument('--doc', '-D',
                        default=None,
                        help='Input audio file. Format: m4a',
                        type=pathlib.Path)
    parser.add_argument('--model', '-M',
                        default=None,
                        help='Input reports',
                        type=pathlib.Path)
    parser.add_argument('--model_input',
                        default=None,
                        help='Input reports',
                        type=pathlib.Path)
    args = parser.parse_args()
    return args


def simplify(s, table):
    for key in table:
        s = s.replace(key, table[key])
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    s = s.decode().strip()
    return "".join(s.split())


def test_X_T(g2p, model, X, Y, f_out):
    track = recording_storage.Recording(X).process(g2p)

    f_logger.debug(f"Hypo orth: {track.hypothesis}")
    f_logger.debug(f"Hypo phon: {track.hypothesis_phon}")

    reference = document_importer.import_document(Y)
    reference = document_importer.preprocess(reference)
    f_logger.debug(f"Reference orth:{reference}")
    reference_phon = g2p.transliterate(reference)
    f_logger.debug(f"Reference phon:{reference_phon}")

    f_logger.info(
        f"WER before fixing: {metrics.wer(reference, track.hypothesis)}")

    max_num_of_kmers = min(100000, len(model.model_kwords))
    f_logger.debug(
        f'Choosing {max_num_of_kmers} from {len(model.model_kwords)} n-grams')
    l1 = random.sample(model.model_kwords, max_num_of_kmers)
    method_simple.test_with_params(track.hypothesis_phon, g2p, l1, track,
                     reference, -1, -0.3, f_out, False, model)


# def main1(args):
#     train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
#         pathlib.Path(".\\data_conf\\mgr\\mama\\wav_files"),
#         pathlib.Path(".\\data_conf\\mgr\\mama\\opisy"),
#         20)

#     g2p = phonemic.G2P()
#     if args.model:
#         model = ngram_model.NGramModel.load(args.model)
#         c_logger.info(f"Using {model.max_n}-gram model")
#     else:
#         # output = document_importer.import_directory(args.model_input)
#         output = document_importer.import_file_list(train_data)
#         # print(output)
#         model = ngram_model.NGramModel(5)
#         c_logger.info(f"Initialising {model.max_n}-gram model")
#         for document in output:
#             model.add_document_skip_short_words(document)
#         model.save(config.TMP_PATH / 'a.pkl')

#     with open(config.TMP_PATH / 'test1.txt', 'w') as f_out:
#         for X, Y in zip(test_data_X, test_data_Y):
#             f_logger.debug(
#                 '-------------------------------------------------------------------------------------')
#             f_logger.info(X.stem)
#             test_X_T(g2p, model, X, Y, f_out)

#     return


def main(args):
    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        pathlib.Path(".\\data_conf\\mgr\\mama\\wav_files"),
        pathlib.Path(".\\data_conf\\mgr\\mama\\opisy"),
        20)

    # ASR = ASR_main_flow.ASR(train_data, 10000)

    g2p = phonemic.G2P()
    if args.model:
        model = ngram_model.NGramModel.load(args.model)
        c_logger.info(f"Using {model.max_n}-gram model")
    else:
        # output = document_importer.import_directory(args.model_input)
        output = document_importer.import_file_list(train_data)
        # print(output)
        model = ngram_model.NGramModel(5)
        c_logger.info(f"Initialising {model.max_n}-gram model")
        for document in output:
            model.add_document_skip_short_words(document)
        model.save(config.TMP_PATH / 'a.pkl')

    with open(config.TMP_PATH / 'test1.txt', 'w') as f_out:
        for X, Y in zip(test_data_X, test_data_Y):
            f_logger.debug(
                '-------------------------------------------------------------------------------------')
            f_logger.info(X.stem)
            test_X_T(g2p, model, X, Y, f_out)

    return



def test_with_params1(dna1, g2p, l1, track, reference, param1, param2, f_out, save_kmer_score=False, model=None):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''

    f_logger.debug(f'param1: {param1}')
    f_logger.debug(f'param2: {param2}')

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

    kgram = tmp_l[0]
    alignment, score = kgram_score[kgram]

    start = alignment.start
    end = alignment.end - alignment.seqA.count('-')
    fixed = kgram.split()
    hypo = dna1

    # forward
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
        candidate = max(list(candidates), key=lambda c: candidates[c].score)
        print(fixed)
        print(candidate)
        print(candidates[candidate])
        fixed.append(candidate)
        # print(candidates[candidate])
        input()
        end += len(candidates[candidate].seqB) - \
            re.search(r'[^-]', candidates[candidate].seqB).start()

    fixed = ' '.join(fixed)

    f_logger.debug("FIXED: {fixed}")
    f_logger.debug(f'WER after fixing: {metrics.wer(reference, fixed)}')


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
    args = parse_args()
    main(args)

    # alignment = pairwise2.align.globalmc(
    #     "AACDFSB"[::-1], "AACX"[::-1], 100, -1000, gap_function, gap_function_no_start_penalty)
    # for a in alignment:
    #     print(pairwise2.format_alignment(*a))
