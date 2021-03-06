import logging
import random
from re import L

import models.ngram_model as ngram_model
import models.ngram_model_phonetic as ngram_model_phonetic
import data_loading.document_importer as document_importer
import phonemic
import mismatch_correction
import data_loading.recording_storage as recording_storage
# import methods.method_simple as method
# import methods.method_advanced as method
# import methods.method_advanced2 as method
# import methods.method_advanced3 as method
# import methods.method_advanced4 as method
# import methods.method_advanced6 as method
# import methods.method_advanced6_simplified as method
# import methods.method_advanced7 as method
# import methods.method_advanced8 as method
# import methods.method_advanced9 as method
# import methods.method_advanced10 as method
# import methods.method_advanced11 as method
# import methods.method_advanced12 as method
import methods.method_advanced13 as method
# import methods.method_advanced13_matrix as method
# import methods.method_advanced14 as method
# import methods.method_advanced15 as method
# import methods.method_simple1 as method
import utils
import config


class ASR:
    def __init__(self, training_data, kmer_limit, load_model):
        self.g2p = phonemic.G2P()

        if load_model:
            self.model = ngram_model_phonetic.NGramModel.load(
                config.MODEL_PATH)
        else:
            self.model = ngram_model_phonetic.NGramModel(self.g2p, 5)
            model_input = document_importer.import_file_list(training_data)
            for document in model_input:
                self.model.add_document(document)
            self.model.save(config.MODEL_PATH)

        self.max_num_of_kmers = min(kmer_limit, len(self.model.model_kwords))
        self.l1 = random.sample(self.model.model_kwords, self.max_num_of_kmers)

    @utils.timeit
    def run(self, sound_file):
        track = recording_storage.Recording(sound_file).process(self.g2p)

        track.hypothesis_phon = mismatch_correction.get_replacements(track.hypothesis_phon)

        fixed = method.test_with_params(
            track.hypothesis_phon,
            self.g2p,
            self.l1,
            track,
            -1,
            -0.3,
            self.model
        )
        return track.hypothesis, fixed
