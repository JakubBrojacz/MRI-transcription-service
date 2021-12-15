import pickle

import config
import data_loading.sound_preprocessing as sound_preprocessing
import julius
import phonemic


class Recording:
    def __init__(self, path):
        self.path = path
        self.save_path = config.TMP_PATH / (self.path.name+'.pkl')

    def process(self, g2p):
        # if self.save_path.exists():
        #     return Recording.load(self.save_path)
        
        wav_file = sound_preprocessing.convert_m4a_wav(self.path, config.TMP_PATH)
        self.hypothesis = julius.run_julius(wav_file)
        self.hypothesis = ' '.join(self.hypothesis[0])
        self.hypothesis_phon = g2p.transliterate(self.hypothesis)
        self.save(self.save_path)
        return self

    @classmethod
    def load(cls, filename):
        print(f'loading rec from {filename}')
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def save(self, filename):
        print(f'saving rec from {filename}')
        with open(filename, 'wb') as f:
            model = pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return model
