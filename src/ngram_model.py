from nltk.util import ngrams


class NGramModel:
    def __init__(self, max_n=5):
        self.max_n = max_n
        self.model = {}

    def add_phrase(self, phrase):
        model_step = self.model
        for word in phrase:
            if word not in model_step:
                model_step[word] = {}
            model_step = model_step[word]
        model_step["NUM"] = model_step.get("NUM", 0) + 1

    def add_document(self, content):
        splitted = content.split()
        for i in range(1, self.max_n+1):
            igrams = ngrams(splitted, i)
            for igram in igrams:
                self.add_phrase(igram)

