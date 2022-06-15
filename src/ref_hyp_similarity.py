import argparse
import pathlib
import random
import logging
import json

from Bio import pairwise2
import matplotlib.pyplot as plt

import config
import data_loading.document_importer as document_importer
import data_loading.train_test_data as train_test_data
import phonemic
import data_loading.recording_storage as recording_storage


logger = logging.getLogger("Main_Console_Logger")
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--moje',
                        action='store_true',
                        help='use recordings from MY_WAV as training samples')
    parser.add_argument('--plot',
                        action='store_true',
                        help='create relevant plots')
    args = parser.parse_args()
    return args


def main(args):
    min_mismatch_to_match_ratio = 3
    min_window_size = 30
    min_seq_size = 20
    plot_path = config.PLOT_PATH / "ref_hyp_similarity"
    plot_path.mkdir(parents=True, exist_ok=True)

    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        config.WAV_ORIGINAL_PATH,
        config.TRANSCIPTIONS_PATH,
        10000,
        moje=args.moje,
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

        with open(config.RULES_PATH, 'w') as f:
            json.dump(result_assignemnts, f, indent=2)

        if args.plot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(Y, label="original", alpha=0.5)
            for (start_id, end_id) in windows:
                ax.axvspan(start_id, end_id, facecolor='g', alpha=0.5)
            plt.savefig(plot_path / f"plot_{X.stem}.png")
            plt.close()

d
if __name__ == '__main__':
    args = parse_args()
    random.seed(1375)
    main(args)
