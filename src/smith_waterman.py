import argparse
import pathlib
import numpy as np
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq1", type=pathlib.Path,
                        help="PATH: first input fasta file",
                        default="seq1.fasta")
    parser.add_argument("--seq2", type=pathlib.Path,
                        help="PATH: second input fasta file",
                        default="seq2.fasta")
    parser.add_argument("--id1", type=str,
                        help="ID: first input id from protein database")
    parser.add_argument("--id2", type=str,
                        help="ID: second input id from protein database")
    parser.add_argument("--config", type=pathlib.Path,
                        help="PATH: config file",
                        default="config.txt")
    return parser.parse_args()


def load_config(file):
    config = {}
    config["GAP PENALTY"] = 0
    config["SAME AWARD"] = 0
    config["DIFFERENCE PENALTY"] = 0
    config["MAX SEQ LENGTH"] = 0
    config["MAX NUMBER PATHS"] = 0
    config["SMITH WATERMAN"] = 0
    with open(file, 'r') as f:
        for line, key in itertools.product(f.readlines(), config):
            if line.startswith(key):
                config[key] = int(line[len(key):])
    return config


def alignment(dna1, dna2, config):
    n1 = len(dna1)
    n2 = len(dna2)
    if n1 > config["MAX SEQ LENGTH"] or n2 > config["MAX SEQ LENGTH"]:
        # print(f'{n1}, {n2}')
        raise Exception("Too long dna")

    V_table = np.zeros((n1+1, n2+1), int)

    if config["SMITH WATERMAN"]:
        V_table[:, 0] = [
            0 for i in range(n1+1)
        ]
        V_table[0, :] = [
            0 for i in range(n2+1)
        ]
    else:
        V_table[:, 0] = [
            config["GAP PENALTY"]*i for i in range(n1+1)
        ]
        V_table[0, :] = [
            config["GAP PENALTY"]*i for i in range(n2+1)
        ]

    for i in range(n1):
        for j in range(n2):
            if dna1[i] == dna2[j]:
                V_table[i+1, j+1] = V_table[i, j] + config["SAME AWARD"]
            else:
                V_table[i+1, j+1] = V_table[i, j] + \
                    config["DIFFERENCE PENALTY"]
            V_table[i+1, j+1] = max(
                V_table[i+1, j+1],
                V_table[i+1, j] + config["GAP PENALTY"],
                V_table[i, j+1] + config["GAP PENALTY"]
            )
            if config["SMITH WATERMAN"]:
                V_table[i+1, j+1] = max(
                    V_table[i+1, j+1],
                    0
                )

    # print(np.array(V_table))
    return V_table


def get_neighbours(vertex, config, V_table, dna1, dna2):
    neighbours = []
    if (
            vertex[0] > 0 and
            vertex[1] > 0 and
            V_table[tuple(vertex)] == V_table[tuple(vertex-[1, 1])] + config["SAME AWARD"] and
            dna1[vertex[0]-1] == dna2[vertex[1]-1]
    ):
        neighbours.append(vertex-[1, 1])
    if (
            vertex[0] > 0 and
            vertex[1] > 0 and
            V_table[tuple(vertex)] == V_table[tuple(vertex-[1, 1])] + config["DIFFERENCE PENALTY"] and
            dna1[vertex[0]-1] != dna2[vertex[1]-1]
    ):
        neighbours.append(vertex-[1, 1])
    if (
            vertex[0] > 0 and
            V_table[tuple(vertex)] == V_table[tuple(
                vertex-[1, 0])] + config["GAP PENALTY"]
    ):
        neighbours.append(vertex-[1, 0])
    if (
            vertex[1] > 0 and
            V_table[tuple(vertex)] == V_table[tuple(
                vertex-[0, 1])] + config["GAP PENALTY"]
    ):
        neighbours.append(vertex-[0, 1])
    return neighbours


def end_condition(vertex, config, V_table):
    if np.array_equal(vertex, np.array([0, 0])):
        return True
    if config["SMITH WATERMAN"] and V_table[tuple(vertex)]==0:
        return True
    return False


def get_entry_points(config, V_table):
    if config["SMITH WATERMAN"]:
        max_value = np.max(V_table)
        return ([
            np.array([i, j])
            for i, j in np.ndindex(V_table.shape)
            if V_table[i,j] == max_value
        ], max_value)
    else:
        return ([np.array(V_table.shape)-1], -100)


def DFS(config, V_table, entry_point, final_paths, dna1, dna2):

    stack = [(entry_point, np.array([-1, -1]))]
    path = []

    while stack:
        vertex, prev = stack.pop()
        # print(vertex)

        while path and not np.array_equal(path[-1], prev):
            path.pop()

        path.append(vertex)

        if end_condition(vertex, config, V_table):
            final_paths.append(path.copy())
            if len(final_paths) >= config["MAX NUMBER PATHS"]:
                break 
            continue

        for neighbour in get_neighbours(vertex, config, V_table, dna1, dna2):
            stack.append((neighbour, vertex))

    # print(final_paths)
    return final_paths


def visualise_generate_spaces(dna, path, ind):

    tmp = [
        (prev[ind],id)
        for id, (prev, nex) in enumerate(zip(path[:-1], path[1:]))
        if prev[ind] != nex[ind]
    ]
    # print(tmp)
    s = ['-' for _ in path[:-1]]
    for dna_id, s_id in tmp:
        s[s_id] = dna[dna_id]

    s = list(dna[:path[0][ind]]) + s
    s = [' ' for _ in range(path[0][1-ind] - path[0][ind])] + s

    s = s + list(dna[path[-1][ind]:])
    # s = s + [' ' for _ in range(path[-1][1-ind] - path[-1][ind])]
    # dont need to write empty spaces at the end of string
    
    return ''.join(s)


def visualise(dna1, dna2, paths):
    line_len = 120
    for path_id, path in enumerate(paths):
        path.reverse()
        s1 = visualise_generate_spaces(dna1, path, 0)
        s2 = visualise_generate_spaces(dna2, path, 1)
        n = max(len(s1), len(s2))

        print(f'--------------------------------------------------------------------')
        print(f'PATH {path_id+1}')
        print(f'--------------------------------------------------------------------')
        print()
        for i in range(n//line_len+1):
            print(s1[i*line_len:(i+1)*line_len-1])
            print(s2[i*line_len:(i+1)*line_len-1])
            print()


if __name__ == "__main__":
    args = parse_args()
    if args.id1:
        labels1, content1 = load_from_id(args.id1)
        labels2, content2 = load_from_id(args.id2)
    else:
        labels1, content1 = load_fasta(args.seq1)
        labels1, content1 = load_fasta(args.seq2)
    config = load_config(args.config)
    print(config)

    dna1 = content1[0]
    dna2 = content2[0]
    print(dna1)
    print(dna2)

    table = alignment(dna1, dna2, config)
    final_paths = []
    entry_points, score = get_entry_points(config, table)
    for entry_point in entry_points:
        final_paths = DFS(config, table, entry_point, final_paths, dna1, dna2)
        if len(final_paths) >= config["MAX NUMBER PATHS"]:
            break 
    visualise(dna1, dna2, final_paths)
