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
    parser.add_argument('--moje',
                        action='store_true',
                        help='Use my recordings as testset')
    parser.add_argument('--model_input',
                        default=None,
                        help='Input reports',
                        type=pathlib.Path)
    parser.add_argument('--test_size', '-T',
                        default=20,
                        help='Input reports',
                        type=int)
    parser.add_argument('--skip', '-S',
                        action='store_true',
                        help='Skip saving logs')
    parser.add_argument('--desc',
                        default="",
                        help='Appended to experiment name in logs',
                        type=str)
    args = parser.parse_args()
    return args


def setup_loggers(args):
    if args.skip:
        experiment_no = 1
    else:
        experiment_no = random.randint(100000, 999999)
    log_path = config.LOG_PATH / f'experiment_{experiment_no}{args.desc}'
    log_path.mkdir(parents=True, exist_ok=True)
    initial_log_message = \
        f"Experiment {experiment_no}: testing {ASR_main_flow.method}\n"\
        f"args: {args}"

    logger = logging.getLogger("Main_File_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        log_path / 'main.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(initial_log_message)

    logger = logging.getLogger("Main_Console_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(initial_log_message)

    logger = logging.getLogger("Time_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        log_path / 'time.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(initial_log_message)

    logger = logging.getLogger("Method_Logger")
    logger.setLevel(logging.DEBUG)
    # logger.handlers = []
    handler = logging.FileHandler(
        log_path / 'method.log', encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(initial_log_message)


def main(args):
    train_data, test_data_X, test_data_Y = train_test_data.get_train_test_data(
        config.WAV_ORIGINAL_PATH,
        config.TRANSCIPTIONS_PATH,
        args.test_size,
        moje=args.moje,
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
    args = parse_args()
    setup_loggers(args)
    random.seed(1375)
    main(args)
