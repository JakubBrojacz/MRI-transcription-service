import logging
import random

import ngram_model
import data_loading.document_importer as document_importer
import phonemic
import data_loading.recording_storage as recording_storage
import method_simple

f_logger = logging.getf_logger("File_Logger")
f_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('spam.log')
fh.setLevel(logging.DEBUG)
f_logger.addHandler(fh)


class ASR:
    def __init__(self, training_data, kmer_limit):
        self.g2p = phonemic.G2P()

        self.model = ngram_model.NGramModel(5)
        model_input = document_importer.import_file_list(training_data)
        for document in model_input:
            self.model.add_document(document)

        self.max_num_of_kmers = min(100000, len(self.model.model_kwords))
        self.l1 = random.sample(self.model.model_kwords, self.max_num_of_kmers)

    def run(self, sound_file):
        track = recording_storage.Recording(sound_file).process(self.g2p)
        hypothesis = track.hypothesis_phon

        fixed = method_simple.run_with_params(hypothesis,
                                              self.g2p,
                                              self.l1,
                                              track,
                                              -1,
                                              -0.3,
                                              self.model)
        return hypothesis, fixed
