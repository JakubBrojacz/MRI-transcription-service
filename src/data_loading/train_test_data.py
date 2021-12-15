import random

import data_loading.document_importer as document_importer


def get_train_test_data(sound_path, doc_path, num_of_test_entries):
    all_data = [file for file in doc_path.iterdir() if file.suffix=='.doc']
    test_data_X = [file for file in sound_path.iterdir() if file.suffix=='.wav']
    test_data_X = random.sample(test_data_X, num_of_test_entries)
    test_data_stems = [file.stem for file in test_data_X]
    test_data_Y = []
    for stem in test_data_stems:
        d = [file for file in all_data if file.stem == stem][0]
        test_data_Y.append(d)
    train_data = [file for file in all_data if file.stem not in test_data_stems]
    return train_data, test_data_X, test_data_Y
