import logging
import itertools

from tqdm import tqdm
from Bio import pairwise2
import numpy as np

import config


# plot of word alignment, method not returning anything


f_logger = logging.getLogger("Method_Logger")


def get_last_word(words, transitions, word_id, column_id, num_of_words):
    fixed_ids = [word_id]

    for trans in transitions[1:column_id][::-1]:
        fixed_ids.append(trans[fixed_ids[-1]])

    # print(fixed_ids)
    last_words = [
        words[idx]
        for idx, _ in itertools.groupby(fixed_ids[::-1])
    ][-num_of_words:]
    return last_words


def test_with_params(dna1, g2p, l1, track, param1, param2, model):
    '''
    param1 - break start (-1)
    param2 - break continue (-0.3)
    '''
    # dna1 = dna1[:40]

    # m1, dna1 = test_utils.prepare_test_data()
    # model.model = m1

    f_logger.info("start method")
    np.random.seed(42)

    words = list(model.reverse_pronounciation_dictionary)
    word_to_id = {
        word: idx
        for idx, word in enumerate(words)
    }

    word_alignment_table = np.zeros((len(dna1), len(words)))

    import matplotlib.pyplot as plt
    plot_path = config.ROOT_PATH / "plots" / "word_local_alignment"
    plot_path.mkdir(parents=True, exist_ok=True)
    for word, word_id in tqdm(word_to_id.items()):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot([max(ord(sign), 80) for sign in dna1], label="original", alpha=0.5)
        alignments = pairwise2.align.localms(
            dna1,
            word,
            10,
            -5,
            -15,
            -15
        )
        for alignment in alignments:
            alignment_len = alignment.end-alignment.start
            if alignment_len > len(word)+3:
                continue
            ax.axvspan(alignment.start, alignment.end, facecolor='g', alpha=0.5)
            for i in range(alignment.start, alignment.end):
                word_alignment_table[i, word_id] = max(
                    word_alignment_table[i, word_id], alignment.score)
        plt.savefig(plot_path / f"word_alignment_{word}.png")
        plt.title(f"Word alignment: {word}")
        plt.xlabel("hypothesis position")
        plt.ylabel("letter")
        ax.get_yaxis().set_visible(False)
        plt.show()
        plt.close()


    # probability_matrix = np.zeros((len(dna1), len(words)))
    # for position in tqdm(list(range(len(dna1)))):
        

    return None


# x is gap position in seq, y is gap length
def gap_function_no_start_penalty(x, y, w1, w2):
    if x == 0:
        return 0
    return gap_function(x, y, w1, w2)


def gap_function(x, y, w1, w2):  # x is gap position in seq, y is gap length
    if y == 0:  # No gap
        return 0
    return w1 + (y-1)*w2
