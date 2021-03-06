import itertools
from Bio import pairwise2
import tqdm
import json
import random
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import argparse

import phonemic
import data_loading.train_test_data as train_test_data
import data_loading.document_importer as document_importer
import data_loading.recording_storage as recording_storage
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--plot',
                        action='store_true',
                        help='create relevant plots')
    args = parser.parse_args()
    return args


def main(args):
    random.seed(0)
    plot_path = config.PLOT_PATH / "conf_table_creator"
    plot_path.mkdir(parents=True, exist_ok=True)

    _, train_data_wav, train_data_doc = train_test_data.get_train_test_data(
        config.ROOT_PATH / ".\\data_conf\\mgr\\mimi\\wav_files_orginal",
        config.ROOT_PATH / ".\\data_conf\\mgr\\mimi\\opisy",
        200,
        moje=False,
        dont=False)

    g2p = phonemic.G2P()

    alignment_score = {}
    signs_set = set()
    all_signs = {}
    signs_assignment = {}

    for document_doc, document_wav in tqdm.tqdm(zip(train_data_doc,
                                                    train_data_wav)):
        if document_doc.suffix not in document_importer.SUPPORTED_TEXT_TYPES:
            continue
        reference_orth = document_importer.import_document(document_doc)
        reference_orth = document_importer.preprocess(reference_orth)
        reference = g2p.transliterate(reference_orth)
        hypothesis_rec = recording_storage.\
            Recording(document_wav)
        if not hypothesis_rec.save_path.exists():
            print(f"Skipping {hypothesis_rec.save_path}")
            continue
        hypothesis = hypothesis_rec.process(g2p).hypothesis_phon

        signs_set.update(reference)
        signs_set.update(hypothesis)

        # if '.' in hypothesis or '.' in reference:
        #     print('wtf')
        alignment = pairwise2.align.globalms(
            reference,
            hypothesis,
            1,
            -1,
            -1.1, -0.5,
            one_alignment_only=True
        )
        alignment = alignment[0]
        # print(pairwise2.format_alignment(*alignment))
        # input()

        # Create f matrix
        for s1, s2 in zip(alignment.seqA, alignment.seqB):
            if '-' in [s1, s2]:
                continue
            all_signs[s1] = all_signs.get(s1, 0) + 1
            all_signs[s2] = all_signs.get(s2, 0) + 1
            if s1 <= s2:
                if s1 not in alignment_score:
                    alignment_score[s1] = {}
                if s2 not in alignment_score[s1]:
                    alignment_score[s1][s2] = 0
                alignment_score[s1][s2] += 1
            else:
                if s2 not in alignment_score:
                    alignment_score[s2] = {}
                if s1 not in alignment_score[s2]:
                    alignment_score[s2][s1] = 0
                alignment_score[s2][s1] += 1

    signs_assignment = {
        sign: chr(ord('A')+i)
        for i, sign in enumerate(signs_set)
    }

    for key in signs_set:
        if key not in all_signs:
            all_signs[key] = 0

    # fill in gaps in f matrix
    for key1, key2 in itertools.product(all_signs, all_signs):
        if key1 not in alignment_score:
            alignment_score[key1] = {}
        if key2 not in alignment_score[key1]:
            alignment_score[key1][key2] = 0

    # create q matrix
    num_of_occurences = sum((
        alignment_score[key][replacement]
        for key in alignment_score
        for replacement in alignment_score[key]
    ))
    for key in alignment_score:
        for replacement in alignment_score[key]:
            alignment_score[key][replacement] = \
                alignment_score[key][replacement] / num_of_occurences

    # create p matrix
    p_matrix = {}
    for key in all_signs:
        p_matrix[key] = all_signs[key] / sum((
            all_signs[key2]
            for key2 in all_signs
        ))

    # create s matrix
    s_matrix = {}
    for key1, key2 in itertools.product(alignment_score, alignment_score):
        if key1 == key2:
            e_12 = p_matrix[key1]*p_matrix[key2]
        else:
            e_12 = 2*p_matrix[key1]*p_matrix[key2]
        odds = max(alignment_score[key1][key2],
                   alignment_score[key2][key1])/e_12
        if odds == 0:
            s_matrix[key1+key2] = -13
        else:
            s_matrix[key1+key2] = math.log2(odds)*2

    with open(config.CONF_TABLE_PATH, 'w') as f:
        json.dump(alignment_score, f, indent=4)

    with open(config.MATRIX_PATH, 'w') as f:
        json.dump(s_matrix, f, indent=4)

    if not args.plot:
        return

    heatmap_axis = sorted(list(alignment_score))
    heatmap_axis.remove('.')
    heatmap_data = np.array([
        [
            s_matrix[
                s1 + s2
            ]
            for s2 in heatmap_axis
        ]
        for s1 in heatmap_axis
    ])
    ax = sns.heatmap(heatmap_data, annot=False, xticklabels=heatmap_axis, yticklabels=heatmap_axis, cmap="PiYG")
    ax.set_title("Similarity matrix")
    # plt.show()
    plt.savefig(plot_path / "similarity_matrix.png")
    plt.close()

    clustermap_data = pd.DataFrame(heatmap_data, columns=heatmap_axis, index=heatmap_axis)
    linkage_matrix = heatmap_data
    linkage_matrix = -linkage_matrix
    linkage_matrix -= np.amin(linkage_matrix)
    for i in range(len(heatmap_axis)):
        linkage_matrix[i][i] = 0
    ax = sns.heatmap(linkage_matrix, annot=False, xticklabels=heatmap_axis, yticklabels=heatmap_axis, cmap="PiYG")
    ax.set_title("Linkage matrix")
    plt.show()
    plt.savefig(plot_path / "linkage_matrix.png")
    plt.close()
    linkage = hc.linkage(sp.distance.squareform(linkage_matrix), method='average')
    g = sns.clustermap(clustermap_data, row_cluster=False, col_linkage=linkage)
    # ax.set_title("Substitution matrix")
    plt.savefig(plot_path / "clustered_phonemes.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
