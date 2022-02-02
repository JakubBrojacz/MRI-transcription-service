import argparse
import pathlib
import random
import sys
import logging


import config
import data_loading.document_importer as document_importer
import metrics
import data_loading.train_test_data as train_test_data
import ASR_main_flow


f_logger = logging.getLogger("Main_File_Logger")
c_logger = logging.getLogger("Main_Console_Logger")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recognise sound')
    parser.add_argument('--input', '-I',
                        default=None,
                        help='Input audio file. Format: m4a',
                        type=pathlib.Path)
    parser.add_argument('--doc', '-D',
                        default=None,
                        help='Input audio file. Format: m4a',
                        type=pathlib.Path)
    parser.add_argument('--model', '-M',
                        action='store_true',
                        help='Input reports')
    parser.add_argument('--model_input',
                        default=None,
                        help='Input reports',
                        type=pathlib.Path)
    args = parser.parse_args()
    return args


def setup_loggers():
    logger = logging.getLogger("Main_File_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        config.LOG_PATH / 'main.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger = logging.getLogger("Main_Console_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger = logging.getLogger("Time_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        config.LOG_PATH / 'time.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger = logging.getLogger("Method_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        config.LOG_PATH / 'method.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def main(args):

    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        pathlib.Path(".\\data_conf\\mgr\\mama\\wav_files"),
        pathlib.Path(".\\data_conf\\mgr\\mama\\opisy"),
        3,
        moje=pathlib.Path(".\\data_conf\\mgr\\moje_nagrania"),
        dont=False)

    ASR = ASR_main_flow.ASR(train_data, 100000, args.model)

    for X, Y in zip(test_data_X, test_data_Y):
        f_logger.info(f'processing: {X.stem}')
        hypothesis, fixed = ASR.run(X)
        reference = document_importer.import_document(Y)
        reference = document_importer.preprocess(reference)
        f_logger.info(f'ref: {reference}')
        f_logger.info(f'hyp: {hypothesis}')
        f_logger.info(f'hyp: {metrics.wer(reference, hypothesis)}')
        f_logger.info(f'fix: {fixed}')
        f_logger.info(f'fix: {metrics.wer(reference, fixed)}')


if __name__ == '__main__':
    setup_loggers()
    args = parse_args()
    random.seed(1375)
    main(args)
