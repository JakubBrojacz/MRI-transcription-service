import subprocess
import pathlib
import sys

import config
from config import JULIUS_PATH, ROOT_PATH


def run_julius(input_wav: pathlib.Path):

    with open(JULIUS_PATH / 'test.dbl', 'w') as f:
        f.write(input_wav)

    process = subprocess.Popen([
        config.JULIUS_EXE_PATH,
        "-dnnconf", config.JULIUS_PATH / 'dnn.jconf',
        "-C", config.JULIUS_PATH / 'julius.jconf',
    ],
        stdout=subprocess.PIPE)
    output = process.stdout.read()

    encoding_type = 'iso8859_2'

    lines = output.splitlines()
    lines = [
        line.decode(encoding_type)
        for line in lines
    ]

    line_prefix = 'sentence1'
    sentences = [
        line[len(f'{line_prefix}: <s> '):-len(' </s>')]
        for line in lines
        if line.startswith(line_prefix)
    ]

    line_prefix_phonemes = 'phseq1'
    phonemes = [
        line[len(f'{line_prefix_phonemes}: '):]
        for line in lines
        if line.startswith(line_prefix_phonemes)
    ]

    # text = '\n'.join(sentences)
    # with open('aa.txt', 'w', encoding=de_type) as f:
    #     f.write(text)

    return sentences, phonemes


if __name__ == "__main__":
    sentences, phonemes = run_julius(JULIUS_PATH/"len.wav", ROOT_PATH/"aa.txt")
    print(sentences)


#  "-input", f"{input_wav}",
#         "-filelist", "test.dbl",
#         "-htkconf", "wav_config",
#         "-h", "PLPL-v7.1.am",
#         "-hlist", "PLPL-v7.1.phn",
#         "-d", "PLPL-v7.1.lm",
#         "-v", "PLPL-v7.1.dct",
#         "-b", "4000",
#         "-lmp", "12", "2",
#         "-lmp2", "12", "2",
#         "-walign",
#         "-fallback1pass",
#         "-multipath",
#         "-iwsp",
#         "-norealtime",
#         "-iwcd1", "max",
#         "-spmodel", "sp",
#         "-spsegment",
#         "-gprune", "none",
#         "-no_ccd",
#         "-sepnum", "150",
#         "-b2", "360",
#         "-n", "40",
#         "-s", "2000",
#         "-m", "8000",
#         "-lookuprange", "5",
#         "-sb", "80",
#         "-forcedict",
#         "-cutsilence",
