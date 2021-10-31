import argparse
import pathlib
import json

import document_importer
import ngram_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--input', '-I',
                        default=None,
                        help='Input reports',
                        type=pathlib.Path)
    args = parser.parse_args()
    return args


def main(args):
    output = document_importer.import_directory(args.input)
    print(output)
    model = ngram_model.NGramModel(5)
    for document in output:
        model.add_document(document)
    print(model.model)
    with open("tmp/a.json", 'w') as f:
        json.dump(model.model, f, indent=4)



if __name__ == '__main__':
    args = parse_args()
    main(args)
