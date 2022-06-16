import random

import config


def get_train_test_data(sound_path, doc_path, num_of_test_entries, moje=False, dont=False):
    if moje:
        test_data_X = [
            file for file in config.MY_WAV.iterdir()
            if file.suffix == '.m4a'
        ]
    else:
        test_data_X = [
            sound_path / file.name.replace('.m4a', '.wav')
            for file in config.MY_WAV.iterdir()
            if file.suffix == '.m4a'
        ]
        if num_of_test_entries > len(test_data_X):
            additional_test = [
                file for file in sound_path.iterdir()
                if file.suffix == '.wav' and file not in test_data_X
            ]
            test_data_X.extend(random.sample(
                additional_test,
                min(num_of_test_entries-len(test_data_X), len(additional_test))
            ))

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
    
    if len(train_data) < 1:
        raise Exception("Empty training data set")

    return train_data, test_data_X, test_data_Y
