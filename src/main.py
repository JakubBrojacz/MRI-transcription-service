import argparse
import pathlib
import unicodedata
import json

import pandas as pd
from tqdm import tqdm
from Bio import pairwise2

import sound_preprocessing
import julius
import config
import ngram_model
import document_importer
import needelman_wunsh
import smith_waterman
import phonemic
import recording_storage
import metrics


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


def main(args):
    g2p = phonemic.G2P()
    if args.model:
        model = ngram_model.NGramModel.load(args.model)
        print(f"Using {model.max_n}-gram model")
    else:
        output = document_importer.import_directory(args.model_input)
        # print(output)
        model = ngram_model.NGramModel(3)
        print(f"Initialising {model.max_n}-gram model")
        for document in output:
            model.add_document(document)
        # model.save(config.TMP_PATH / 'a.pkl')

    track = recording_storage.Recording(args.input).process(g2p)

    # model = ngram_model.NGramModel.from_json(config.ROOT_PATH/'tmp'/'a.json')

    print()
    print("Hypo orth:")
    print(track.hypothesis)
    print()
    print("Hypo phon:")
    print(track.hypothesis_phon)

    reference = document_importer.import_document(args.doc)
    reference = document_importer.preprocess(reference)
    print()
    print("reference orth:")
    print(reference)
    reference_phon = g2p.transliterate(reference)
    print()
    print("reference phon:")
    print(reference_phon)

    print()
    print(f"WER: {metrics.wer(reference, track.hypothesis)}")

    print()
    print("VISUALISATION!!!")

    config_nw = {}
    config_nw["GAP PENALTY"] = -1
    config_nw["SAME AWARD"] = 3
    config_nw["DIFFERENCE PENALTY"] = -1
    config_nw["MAX SEQ LENGTH"] = 10000
    config_nw["MAX NUMBER PATHS"] = 1
    config_nw["SMITH WATERMAN"] = 1
    # table = needelman_wunsh.alignment(track.hypothesis_phon, reference_phon, config_nw)
    # print(table)
    # paths = needelman_wunsh.DFS(track.hypothesis_phon, reference_phon, config_nw, table)
    # needelman_wunsh.visualise(track.hypothesis_phon, reference_phon, paths)

    # aa = "KTÓRE NIESTETY BYŁY BYDŁEM CO BYŁEGO MINISTRA BY M.IN." +\
    #     " SPECYFICZNE TŁUMACZYŁ PREMIER WIELKOŚCI GOOGLE CZTERY" +\
    #     " METRY KLIKAJĄC ZAPEWNE SEJM SUBSTANCJI ŻEBY BYŁY USŁUGĘ" +\
    #     " PRAWIDŁOWY TO CO ROBIMY PRZEMIESZCZAMY SYMETRYCZNIE W" +\
    #     " POSZERZONYM O TO ŻEBY TYLKO KOMOROWYCH KROPKĘ E NIE MA " +\
    #     "PRZYGODĘ PRZEŻYŁ CZYSTE SZEROKOŚCI OK. CZTERY I PÓŁ METRYKĘ" +\
    #     " A NA CZWARTYM MIEJSCU W KTÓRYM BYŁY PRAWIDŁOWE NIE SKŁADA SI" +\
    #     "Ę Z NIEPRAWIDŁOWEGO PROWADZENIA KONTRASTU PROBLEM W SZERSZYM " +\
    #     "MIŁOŚĆ DO TORBY WIELKOŚCI RODZĄCEGO SZEŚCIU MILIMETRÓW NIEZMI" +\
    #     "ERNIE WYJAŚNIONO MĘSKIM CZY KILKA DROBNEGO ZGODNIE Z TYŁU NAC" +\
    #     "ZEPY RP TO WSZYSTKO"
    # epi = epitran.Epitran('pol-Latn')
    # output_phon = epi.transliterate(aa)

    kgram_score = {}
    # track.hypothesis_phon = 'meskjimtsIkjilkadrobnegozgodneztIwunatsepIrp'
    dna1 = track.hypothesis_phon
    import random
    l1 = random.sample(model.model_kwords, min(10000, len(model.model_kwords)))
    for kgram in tqdm(l1):
        dna2 = g2p.transliterate(''.join(kgram.split()))
        alignemnt = pairwise2.align.localxs(
            dna1, dna2, -1, -0.3, one_alignment_only=True)[0]
        # print()
        # print(pairwise2.format_alignment(*alignemnt))
        # print(alignemnt)
        # input()
        score = alignemnt.score / len(dna2)

        kgram_score[kgram] = (alignemnt, score)

        # if score / len(kgram) > 1.2:
        #     print('', file=f)
        #     print(kgram, file=f)
        # print()
        # print(kgram)
        # print(score / len(kgram))

    tmp_l = list(l1)
    tmp_l.sort(key=lambda x: kgram_score[x][1], reverse=True)
    with open(config.TMP_PATH / 'output.txt', 'w', encoding='utf-8') as f:
        for kgram in tmp_l:
            print(f'{kgram}\t\t{kgram_score[kgram][1]}', file=f)
        # json.dump(tmp_l, f, indent=4)

    from itertools import groupby

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

    used_kgrams = [""]
    last_used = 1
    hypo = track.hypothesis_phon
    ids = [0 for sign in hypo]
    start, end = 0, 0
    for kgram in tqdm(tmp_l[:1000]):
        # print(ids[start:end])
        alignment, score = kgram_score[kgram]
        start = alignment.start
        end = alignment.end - alignment.seqA.count('-')

        # print()
        # print(kgram)
        # print(ids[start:end])
        # print(' '.join([
        #         used_kgrams[kgram_id]
        #         for kgram_id, _ in groupby(ids[start:end])
        #     ]))
        if any(ids[start:end]):
            # continue
            if ids[start] and ids[end-1]:
                continue
            tmp_kgram = (' '.join([
                used_kgrams[kgram_id]
                for kgram_id, _ in groupby(ids[start:end])
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
            # print(pairwise2.format_alignment(*alignment))
            # s0 = -1
            # for i, kgram_id in enumerate(ids):
            #     if kgram_id == 0:
            #         print(hypo[i], end='')
            #     else:
            #         if s0 != kgram_id:
            #             s0 = kgram_id
            #             print('-'+used_kgrams[kgram_id], end='-')
            # input()
            continue
        ids[start:end] = [last_used for _ in range(end-start)]
        used_kgrams.append(kgram)
        last_used += 1
        # print(pairwise2.format_alignment(*alignment))
        # s0 = -1
        # for i, kgram_id in enumerate(ids):
        #     if kgram_id == 0:
        #         print(hypo[i], end='')
        #     else:
        #         if s0 != kgram_id:
        #             s0 = kgram_id
        #             print('-'+used_kgrams[kgram_id], end='-')
        # input()

    # print(ids)

    fixed = ' '.join(
        used_kgrams[kgram_id]
        for kgram_id, _ in groupby(ids)
    )
    print()
    print("FIXED:")
    print(fixed)
    print()
    print(f'WER: {metrics.wer(reference, fixed)}')



if __name__ == '__main__':
    args = parse_args()
    main(args)
