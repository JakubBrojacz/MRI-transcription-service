from nltk.util import ngrams
import nltk
import json
import pickle
import numpy as np

import config


class NGramModel:
    def __init__(self, g2p, max_n=5):
        self.max_n = max_n
        self.model = {}
        self.model_kwords = set()
        self.word_list = []
        self.word_to_id = {}
        self.g2p = g2p
        self.pronounciation_dictionary = {}
        self.reverse_pronounciation_dictionary = {}

    @classmethod
    def load(cls, filename):
        print(f'loading model from {filename}')
        with open(config.TMP_PATH / filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def save(self, filename):
        print(f'saving model from {filename}')
        with open(config.TMP_PATH / filename, 'wb') as f:
            model = pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return model

    def add_phrase(self, phrase):
        model_step = self.model
        for word in phrase:

            # G2P
            if word not in self.pronounciation_dictionary:
                word_phon = self.g2p.transliterate(word)
                self.pronounciation_dictionary[word] = word_phon
                self.reverse_pronounciation_dictionary[word_phon] = word
            word = self.pronounciation_dictionary[word]

            if word not in model_step:
                model_step[word] = {}
            model_step = model_step[word]
        model_step["NUM"] = model_step.get("NUM", 0) + 1

    def add_document(self, content):
        splitted = content.split()
        splitted = ['<START>'] + splitted
        for i in range(1, self.max_n+1):
            igrams = ngrams(splitted, i)
            for igram in igrams:
                self.add_phrase(igram)
        kgrams = ngrams(splitted, self.max_n)
        for kgram in kgrams:
            self.model_kwords.add(' '.join(kgram))

    def compile(self):
        self.word_list = list(self.model)
        self.word_to_id = {
            word: idx
            for idx, word in enumerate(self.word_list)
        }

    def predict(self, words):
        result = {}
        for i in range(1, self.max_n):
            if len(words) < i:
                return result
            tmp_model = self.model
            for j in range(i):
                if words[-i+j] not in tmp_model:
                    tmp_model = {}
                    break
                tmp_model = tmp_model[words[-i+j]]
            for pred_word in tmp_model:
                if pred_word == 'NUM':
                    continue
                result[pred_word] = result.get(
                    pred_word, 0) + (tmp_model[pred_word]['NUM']+i)*i*i
        return result

    def ngram_dict(self, words):
        tmp_model = self.model
        for w in range(words):
            if w not in tmp_model:
                return None
            tmp_model = tmp_model[w]
        return tmp_model


    def get_probability(self, precedings, word):
        tmp_model = self.model
        for tmp_w in precedings.split()[-4:]:
            tmp_model = tmp_model.get(tmp_w, {})
        if word not in tmp_model:
            return 0
        sum_prob = sum((
            tmp_model[tmp_w]["NUM"]
            for tmp_w in tmp_model
            if tmp_w != "NUM"
        ))
        return 0.02 + (tmp_model[word]["NUM"]/sum_prob)


    # def predict_backoff(self, words):
    #     result = {}

        

    #     result_array = np.array([
    #         1 for i in self.model
    #     ])
    #     for word in result:

    #     return result
