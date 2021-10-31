import argparse
import pathlib

import sound_preprocessing
import julius
import config


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
    print(output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
