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


X = []
Y = []
hyps = []
with open(ROOT_PATH / f'{args.input}/main.log', encoding='utf-8') as f:
    for line in f:
        if line.startswith('hyp: ') and line[5].isdigit():
            X.append(line[5:-1])
        if line.startswith('hyp: ') and not line[5].isdigit():
            hyps.append(line[5:-1])
        if line.startswith('fix: ') and line[5].isdigit():
            Y.append(line[5:-1])

print(X)
print(Y)


# plt.plot(sorted([float(y)-float(x) for x,y in zip(X,Y)]), '.', label="Experiment")
# plt.show()

X = np.array([float(x) for x in X])
Y = np.array([float(x) for x in Y])
print('hyp')
print('mean')
print(np.mean(X))
print('std')
print(np.std(X))
print('fix')
print('mean')
print(np.mean(Y))
print('std')
print(np.std(Y))
print('diff')
print('mean')
print(np.mean(Y-X))
print('std')
print(np.std(Y-X))


def draw_plot(x1, title, xlabel, step):
    min_1 = math.floor(min(x1)*(1/step))/(1/step)
    max_1 = math.ceil(max(x1)*(1/step))/(1/step)
    print(min_1)
    print(max_1)
    bins = int((max_1 - min_1)/step)

    plt.hist(x1, range = [min_1, max_1], bins=bins,
        ec='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of testcases")
    plt.show()


draw_plot([
    x
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"WER of original ASR hypothesis{args.desc}",
    "WER(hypothesis)",
    0.025)

draw_plot([
    y
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"WER of fixed output of post-processing system{args.desc}",
    "WER(fixed)",
    0.1)

draw_plot([
    y-x
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"Change in WER between ASR hypothesis and fixed output{args.desc}",
    "WER(fixed) - WER(hypothesis)",
    0.1)


# plt.hist(sorted([
#     x
#     for x, y, h in zip(X, Y, hyps)
#     if len(h) > 200]), range = [min_1, max_1], bins=bins)
# plt.title("WER of original ASR hypothesis")
# plt.xlabel("WER(hypothesis)")
# plt.ylabel("Number of testcases")
# plt.show()


# min_1 = math.floor(min(y1)*50)/50
# max_1 = math.ceil(max(y1)*50)/50
# bins = int((max_1 - min_1)/0.02)+1

# plt.hist(sorted([
#     y-0.05
#     for x, y, h in zip(X, Y, hyps)
#     if len(h) > 200]), range = [min_1, max_1], bins=bins)
# plt.title("WER of fixed output of post-processing system")
# plt.xlabel("WER(hypothesis)")
# plt.ylabel("Number of testcases")
# plt.show()


# min_1 = math.floor(min(d1)*50)/50
# max_1 = math.ceil(max(d1)*50)/50
# bins = int((max_1 - min_1)/0.02)+1

# plt.hist(sorted([
#     y-x-0.05
#     for x, y, h in zip(X, Y, hyps)
#     if len(h) > 200]), range = [min_1, max_1], bins=bins)
# plt.title("Change in WER between ASR hypothesis and fixed output")
# plt.xlabel("WER(fixed) - WER(hypothesis)")
# plt.ylabel("Number of testcases")
# plt.show()

# print([
#     y-x-0.05
#     for x, y, h in zip(X, Y, hyps)
#     if len(h) > 200])