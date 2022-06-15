import pathlib


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

TMP_PATH = ROOT_PATH / 'tmp'
LOG_PATH = ROOT_PATH / 'logs'
PLOT_PATH = ROOT_PATH / 'plots'

WAV_ORIGINAL_PATH = pathlib.Path(".\\data_conf\\mgr\\mimi\\wav_files")
TRANSCIPTIONS_PATH = pathlib.Path(".\\data_conf\\mgr\\mimi\\opisy")
MY_WAV = pathlib.Path(".\\data_conf\\mgr\\moje_nagrania")

MODEL_PATH = TMP_PATH / 'model.pkl'
RULES_PATH = TMP_PATH / 'rules.json'
MATRIX_PATH = TMP_PATH / 'confusion_matrix.json'
CONF_TABLE_PATH = TMP_PATH / 'confusion_table.json'


JULIUS_PATH = ROOT_PATH / 'julius'
JULIUS_EXE_PATH = JULIUS_PATH / 'julius-dnn.exe'


FFMPEG_PATH = ROOT_PATH / "ffmpeg" / "ffmpeg.exe"
FFPROBE_PATH = ROOT_PATH / "ffmpeg" / "ffprobe.exe"

