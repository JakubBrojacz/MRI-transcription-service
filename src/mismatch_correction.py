import json
import logging
from Bio import pairwise2

import config

f_logger = logging.getLogger("Method_Logger")


def get_replacements(dna1):
    if not config.RULES_PATH.exists():
        return dna1
        
    with open(config.RULES_PATH) as f:
        replacements = json.load(f)
    for repl_input, repl_output in sorted(replacements, key=lambda x: len(x[0]), reverse=True):
        alignment = pairwise2.align.localms(
            dna1,
            repl_input,
            1,
            -1,
            -1,
            -1,
            one_alignment_only=True
        )[0]
        if alignment.score < len(repl_input)*9.0/10.0:
            continue
        alignment_end = alignment.end - len([
            sign
            for sign in alignment.seqA
            if sign == '-'
        ])
        f_logger.info("Replacement found!")
        f_logger.info(repl_input)
        f_logger.info(repl_output)
        dna1 = dna1[:alignment.start] + repl_output + dna1[alignment_end:]

    return dna1
