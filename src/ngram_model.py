from nltk.util import ngrams
import nltk
import json
import pickle
import numpy as np

import config


class NGramModel:
    def __init__(self, max_n=5):
        self.max_n = max_n
        self.model = {}
        self.model_kwords = set()
        self.word_list = []
        self.word_to_id = {}

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

    def add_document_merge_short_words(self, content):
        splitted = []
        i = 0  # index of sign in content
        j = 0  # index of last sign copied to splitted
        min_entity_len = 7  # merge words with lower number of signs than this
        # print(content)
        # input()
        while i < len(content) and i >= 0:
            i += min_entity_len
            i = content.find(' ', i)
            if i == -1:
                i = len(content)
            splitted.append(content[j:i])
            j = i+1
        # splitted = content.split()
        # splitted = ['<START>' for i in range(self.max_n)] + splitted
        for i in range(1, self.max_n+1):
            igrams = ngrams(splitted, i)
            for igram in igrams:
                self.add_phrase(igram)
        kgrams = ngrams(splitted, self.max_n)
        for kgram in kgrams:
            self.model_kwords.add(' '.join(kgram))

    def add_document_skip_short_words(self, content):
        splitted = content.split()
        min_entity_len = 7  # skip words with lower number of signs than this
        splitted = [word for word in splitted if len(word) >= min_entity_len]
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

    # def predict_backoff(self, words):
    #     result = {}

        

    #     result_array = np.array([
    #         1 for i in self.model
    #     ])
    #     for word in result:

    #     return result
