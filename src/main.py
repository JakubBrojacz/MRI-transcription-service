import argparse
import pathlib
import epitran

import sound_preprocessing
import julius
import config
import ngram_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--input', '-I',
                        default=None,
                        help='Input audio file. Format: m4a',
                        type=pathlib.Path)
    args = parser.parse_args()
    return args


def main(args):
    wav_file = sound_preprocessing.convert_m4a_wav(args.input, config.TMP_PATH)
    output = julius.run_julius(wav_file)
    output = ' '.join(output[0])
    print(output)

    # model = ngram_model.NGramModel.from_json(config.ROOT_PATH/'tmp'/'a.json')

    epi = epitran.Epitran('pol-Latn')
    output_phon = epi.transliterate(output)
    print(output_phon)
    litwo = "Litwo, Ojczyzno moja! ty jesteś jak zdrowie;"+\
    "Ile cię trzeba cenić, ten tylko się dowie,"+\
    "Kto cię stracił. Dziś piękność twą w całej ozdobie"+\
    "Widzę i opisuję, bo tęsknię po tobie."
    print(litwo)
    print(epi.transliterate(litwo))
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
