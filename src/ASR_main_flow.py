import logging
import random

import ngram_model
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage
import method_simple
import method_advanced
import method_advanced2
import method_advanced3
import method_advanced4
import utils


class ASR:
    def __init__(self, training_data, kmer_limit):
        self.g2p = phonemic.G2P()

        self.model = ngram_model.NGramModel(5)
        model_input = document_importer.import_file_list(training_data)
        for document in model_input:
            self.model.add_document(document)

        self.max_num_of_kmers = min(100000, len(self.model.model_kwords))
        self.l1 = random.sample(self.model.model_kwords, self.max_num_of_kmers)

    @utils.timeit
    def run(self, sound_file):
        track = recording_storage.Recording(sound_file).process(self.g2p)

        fixed = method_advanced4.test_with_params(track.hypothesis_phon,
                                                  self.g2p,
                                                  self.l1,
                                                  track,
                                                  -1,
                                                  -0.3,
                                                  self.model)
        return track.hypothesis, fixed
