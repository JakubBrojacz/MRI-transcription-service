import logging
import matplotlib.pyplot as plt
import numpy as np
import math

X = []
Y = []
hyps = []
with open('logs/main_simple.log') as f:
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
print(np.mean(Y-0.05))
print('std')
print(np.std(Y-0.05))
print('diff')
print('mean')
print(np.mean(Y-0.05-X))
print('std')
print(np.std(Y-0.05-X))


def draw_plot(x1, title, xlabel, step):
    min_1 = math.floor(min(x1)*(1/step))/(1/step)
    max_1 = math.ceil(max(x1)*(1/step))/(1/step)
    bins = int((max_1 - min_1)/step)+1
    print(min_1)

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
    "WER of original ASR hypothesis",
    "WER(hypothesis)",
    0.025)

draw_plot([
    y-0.05
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    "WER of fixed output of post-processing system",
    "WER(fixed)",
    0.1)

draw_plot([
    y-0.05-x
    for x, y, h in zip(X, Y, hyps)
    if len(h) > 200],
    "Change in WER between ASR hypothesis and fixed output",
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