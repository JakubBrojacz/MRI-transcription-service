import unicodedata

import epitran
import pandas as pd

import config


class G2P:
    def __init__(self):
        df = pd.read_csv(config.PHONETIC_SIMPLIFICATION_TABLE, sep='\t',
                         header=None, engine='python', encoding="utf-8")
        self.simplification_table = {
            row[1][1]: row[1][6][0]
            for row in df.iterrows()
        }
        self.epi = epitran.Epitran('pol-Latn')

    def preprocess(self, s):
        return s.replace('<START>', '')

    def simplify(self, s):
        for key in self.simplification_table:
            s = s.replace(key, self.simplification_table[key])
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
        s = s.decode().strip()
        return "".join(s.split())

    def transliterate(self, s):
        preprocessed = self.preprocess(s)
        output_phon = self.epi.transliterate(preprocessed)
        return self.simplify(output_phon)
