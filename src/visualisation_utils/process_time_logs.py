import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import pathlib
import argparse


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent.parent / 'logs'


parser = argparse.ArgumentParser(
    description='Recognise sound')
parser.add_argument('--input', '-I',
                    default=None,
                    help='Input log dir name.',
                    type=str)
parser.add_argument('--desc', '-D',
                    default="",
                    help='Input log dir name.',
                    type=str)
args = parser.parse_args()


Y = []
hyps = []
with open(ROOT_PATH / f'{args.input}/time.log', encoding='utf-8') as f:
    for line in f:
        if line.startswith('function:run'):
            Y.append(float(line.split()[-2]))

print(Y)


def draw_plot(x1, title, xlabel):

    plt.hist(x1,
             ec='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of testcases")
    plt.show()


draw_plot(Y,
          f"Execution time of method",
          "time [s]")
