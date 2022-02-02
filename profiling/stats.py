import pstats
from pstats import SortKey

'''
python -m cProfile -o profile_stats .\src\main.py --model a.pkl
'''

with open('profiling/stats_readable.txt', 'w') as stream:
    p = pstats.Stats('profiling/profile_stats', stream=stream)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

