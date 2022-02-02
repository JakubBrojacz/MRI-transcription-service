import logging
import random

import ngram_model
import ngram_model_phonetic
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage
import method_simple
import method_advanced
import method_advanced2
import method_advanced3
import method_advanced4
import method_advanced6
import utils
import config


class ASR:
    def __init__(self, training_data, kmer_limit, load_model):
        self.g2p = phonemic.G2P()

        if load_model:
            self.model = ngram_model_phonetic.NGramModel.load(config.MODEL_PATH)
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

        fixed = method_advanced6.test_with_params(track.hypothesis_phon,
                                                  self.g2p,
                                                  self.l1,
                                                  track,
                                                  -1,
                                                  -0.3,
                                                  self.model)
        return track.hypothesis, fixed
