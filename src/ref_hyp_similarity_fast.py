import argparse
from hashlib import sha3_224
import pathlib
import random
import sys
import logging
import sys
import json

from Bio import pairwise2
import matplotlib.pyplot as plt
import numpy as np
import math

import config
import data_loading.document_importer as document_importer
import metrics
import data_loading.train_test_data as train_test_data
import ASR_main_flow
import models.ngram_model as ngram_model
import models.ngram_model_phonetic as ngram_model_phonetic
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage


logger = logging.getLogger("Main_Console_Logger")
logger.setLevel(logging.DEBUG)


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--moje',
                        action='store_true',
                        help='Use my recordings as testset')
    args = parser.parse_args()
    return args


def main(args):
    if args.moje:
        moje_path = ROOT_PATH / "data_conf\\mgr\\moje_nagrania"
    else:
        moje_path = None

    min_mismatch_to_match_ratio = 3
    min_window_size = 30
    min_seq_size = 20
    plot_path = ROOT_PATH / "plots" / "ref_hyp_similarity"
    plot_path.mkdir(parents=True, exist_ok=True)

    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        ROOT_PATH / "data_conf\\mgr\\mama\\wav_files",
        ROOT_PATH / "data_conf\\mgr\\mama\\opisy",
        10000,
        moje=moje_path,
        dont=False)

    g2p = phonemic.G2P()

    result_assignemnts = []

    for X, Y in zip(test_data_X, test_data_Y):
        logger.info(f'processing: {X.stem}')
        hyp_phon = recording_storage.Recording(X).process(g2p).hypothesis_phon
        reference = document_importer.import_document(Y)
        reference = document_importer.preprocess(reference)
        ref_phon = g2p.transliterate(reference)
        logger.info(f'ref: {ref_phon}')
        logger.info(f'hyp: {hyp_phon}')
        alignment = pairwise2.align.globalms(
            hyp_phon,
            ref_phon,
            1,
            -3,
            -2,
            -1,
            one_alignment_only=True
        )[0]

        Y = [
            min_mismatch_to_match_ratio
            if s1 == s2
            else -1
            for s1, s2 in zip(alignment.seqA, alignment.seqB)
        ]
        windows = []
        start_id = None
        act_window_min = None
        act_window_argmin = None
        for id, y_val in enumerate(Y):
            if start_id is None:
                start_id = id
                window_counter = 0
                act_window_min = 0
                act_window_argmin = id
            window_counter += y_val
            if id-start_id > min_window_size and window_counter < act_window_min:
                act_window_min = window_counter
                act_window_argmin = id
            if window_counter >= 0 or id == len(Y)-1:
                if act_window_min < 0:
                    windows.append((start_id, act_window_argmin))
                start_id = None

        # skip if whole track is weakly recognised
        if sum((
            end_id - start_id
            for (start_id, end_id) in windows
        )) > len(Y)/2:
            continue

        for (start_id, end_id) in windows:
            seqA_part = alignment.seqA[start_id:end_id+1].replace('-', '')
            seqB_part = alignment.seqB[start_id:end_id+1].replace('-', '')
            if len(seqA_part) > min_seq_size and len(seqB_part) > min_seq_size:
                result_assignemnts.append((
                    seqA_part,
                    seqB_part
                ))

        with open(ROOT_PATH / 'tmp.json', 'w') as f:
            json.dump(result_assignemnts, f, indent=2)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(Y, label="original", alpha=0.5)
        for (start_id, end_id) in windows:
            ax.axvspan(start_id, end_id, facecolor='g', alpha=0.5)
        plt.savefig(plot_path / f"plot_{X.stem}.png")
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    random.seed(1375)
    main(args)
