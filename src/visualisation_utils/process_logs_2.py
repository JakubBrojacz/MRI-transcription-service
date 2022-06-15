import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import pathlib
import argparse
import jiwer


def wer(ref, hypo):
    return jiwer.wer(ref, hypo)


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent.parent / 'logs'
PLOT_PATH = ROOT_PATH.parent / 'plots' / 'process_logs'
PLOT_PATH.mkdir(exist_ok=True, parents=True)


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
parser.add_argument('--save_prefix', '-S',
                    default="",
                    help='Input log dir name.',
                    type=str)
args = parser.parse_args()


X = []
Y = []
hyps = []
refs = []
fixes = []
with open(ROOT_PATH / f'{args.input}/main.log', encoding='utf-8') as f:
    for line in f:
        
        if line.startswith('hyp: ') and line[5].isdigit():
            X.append(line[5:-1])
        if line.startswith('hyp: ') and not line[5].isdigit():
            hyps.append(line[5:-1])
        if line.startswith('fix: ') and line[5].isdigit():
            Y.append(line[5:-1])
        if line.startswith('fix: ') and not line[5].isdigit():
            fixes.append(line[5:-1])
        if line.startswith('ref: ') and not line[5].isdigit():
            refs.append(line[5:-1])

print(X)
print(Y)

Y = [
    wer(' '.join(ref.split()[:len(ref.split())//4]), ' '.join(fixa.split()[:len(ref.split())//4]))
    for ref, fixa in zip(refs, fixes)
]

# plt.plot(sorted([float(y)-float(x) for x,y in zip(X,Y)]), '.', label="Experiment")
# plt.show()

X = np.array([float(x) for x in X])
Y = np.array([float(x) for x in Y])
if args.save_prefix:
    with open(PLOT_PATH / f"{args.save_prefix}.txt", 'w') as f:
        print('hyp', file=f)
        print('mean', file=f)
        print(np.mean(X), file=f)
        print('std', file=f)
        print(np.std(X), file=f)
        print('fix', file=f)
        print('mean', file=f)
        print(np.mean(Y), file=f)
        print('std', file=f)
        print(np.std(Y), file=f)
        print('diff', file=f)
        print('mean', file=f)
        print(np.mean(Y-X), file=f)
        print('std', file=f)
        print(np.std(Y-X), file=f)
else:
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


def draw_plot(x1, title, xlabel, step, save_prefix):
    min_1 = math.floor(min(x1)*(1/step))/(1/step)
    max_1 = math.ceil(max(x1)*(1/step))/(1/step)
    print(min_1)
    print(max_1)
    bins = int((max_1 - min_1)/step)

    plt.hist(x1, range=[min_1, max_1], bins=bins,
             ec='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of testcases")
    if save_prefix:
        plt.savefig(PLOT_PATH / f"{save_prefix}{title}.png")
    else:
        plt.show()
    plt.close()


draw_plot([
    x
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"WER of original ASR hypothesis{args.desc}",
    "WER(hypothesis)",
    0.025,
    args.save_prefix)

draw_plot([
    y
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"WER of fixed output of post-processing system{args.desc}",
    "WER(fixed)",
    0.05,
    args.save_prefix)

draw_plot([
    y-x
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    f"Change in WER between ASR hypothesis and fixed output{args.desc}",
    "WER(fixed) - WER(hypothesis)",
    0.05,
    args.save_prefix)


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
