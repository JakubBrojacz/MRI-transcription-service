import random

import data_loading.document_importer as document_importer


def get_train_test_data(sound_path, doc_path, num_of_test_entries, moje=None, dont=False):
    if moje is not None:
        test_data_X = [file for file in moje.iterdir()
                       if file.suffix == '.m4a']
    else:
        test_data_X = [file for file in sound_path.iterdir()
                       if file.suffix == '.wav']
    all_data = [file for file in doc_path.iterdir() if file.suffix == '.doc']
    test_data_X = random.sample(test_data_X, min(
        num_of_test_entries, len(test_data_X)))
    test_data_stems = [file.stem for file in test_data_X]
    test_data_Y = []
    for stem in test_data_stems:
        d = [file for file in all_data if file.stem == stem][0]
        test_data_Y.append(d)
    train_data = [
        file for file in all_data if file.stem not in test_data_stems]
    if dont:
        train_data = [test_data_Y[0]]
        test_data_X = [test_data_X[0]]
        test_data_Y = [test_data_Y[0]]
    return train_data, test_data_X, test_data_Y


