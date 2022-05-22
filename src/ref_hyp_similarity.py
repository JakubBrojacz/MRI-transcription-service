import argparse
from hashlib import sha3_224
import pathlib
import random
import sys
import logging
import sys

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

import config


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

    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        ROOT_PATH / "data_conf\\mgr\\mimi\\wav_files",
        ROOT_PATH / "data_conf\\mgr\\mimi\\opisy",
        10000,
        moje=moje_path,
        dont=False)

    g2p = phonemic.G2P()

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
            -1,
            -1,
            -1,
            one_alignment_only=True
        )[0]
        Y = [
            1
            if s1 == s2
            else 0
            for s1, s2 in zip(alignment.seqA, alignment.seqB)
        ]
        window = 10
        Y1 = []
        for i in range(len(alignment.seqA)):
            min_i = max(0, i-window)
            max_i = min(len(alignment.seqA), i+window)
            val = sum((
                s1 == s2
                for s1, s2 in zip(alignment.seqA[min_i:max_i], alignment.seqB[min_i:max_i])
            )) / (max_i-min_i)
            Y1.append(val)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(Y, label="single letter", alpha=0.5)
        ax.plot(Y1, label="10-letter window")
        quant = np.quantile(Y1, 0.1)
        # plt.hist(Y1, bins=2*window)
        # plt.title(f'0.2 quantile: {quant}')
        # plt.show()

        spans = [0 for _ in range(len(alignment.seqA))]
        min_y2_window = 20
        for i_start in range(len(alignment.seqA)):
            for i_end in range(i_start+min_y2_window, len(alignment.seqA)):
                start_end_similarity = sum((
                    s1 == s2
                    for s1, s2 in zip(alignment.seqA[i_start:i_end], alignment.seqB[i_start:i_end])
                )) / (i_end-i_start)
                if start_end_similarity < quant:
                    for i in range(i_start, i_end):
                        spans[i] = 1
        span_start = None
        for i, s in enumerate(spans):
            if s == 0 and span_start:
                ax.axvspan(span_start, i, facecolor='g', alpha=0.5)
                print(alignment.seqA[span_start:i])
                print(alignment.seqB[span_start:i])
                span_start = None
            if s == 1 and not span_start:
                span_start = i
        if span_start:
            ax.axvspan(span_start, len(spans), facecolor='g', alpha=0.5)
        # ax.plot(Y2, label="mismatches")
        plt.title("Similarity between reference and hypothesis")
        plt.xlabel("Position")
        plt.ylabel("Similarity")
        plt.legend()
        plt.savefig(config.PLOT_PATH / 'ref_hyp_similarity' / f"advanced_plot_{X.stem}.png")
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    random.seed(1375)
    main(args)
