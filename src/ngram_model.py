from nltk.util import ngrams
import json


class NGramModel:
    def __init__(self, max_n=5):
        self.max_n = max_n
        self.model = {}

    @classmethod
    def from_json(cls, json_path, max_n=5):
        print(f'loading model from {json_path}')
        model = NGramModel(max_n)
        with open(json_path, 'r') as f:
            model.model = json.load(f)
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
        splitted = ['<START>' for i in range(self.max_n)] + splitted
        for i in range(1, self.max_n+1):
            igrams = ngrams(splitted, i)
            for igram in igrams:
                self.add_phrase(igram)

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
                result[pred_word] = result.get(pred_word, 0) + tmp_model[pred_word]['NUM']*i
        return result
