import pathlib


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

TMP_PATH = ROOT_PATH / 'tmp'
LOG_PATH = ROOT_PATH / 'logs'
PLOT_PATH = ROOT_PATH / 'plots'

# WAV_ORIGINAL_PATH = ROOT_PATH / ".\\data_conf\\mgr\\mimi\\wav_files"
# TRANSCIPTIONS_PATH = ROOT_PATH / ".\\data_conf\\mgr\\mimi\\opisy"
# MY_WAV = ROOT_PATH / ".\\data_conf\\mgr\\moje_nagrania"
WAV_ORIGINAL_PATH = ROOT_PATH / 'data' / 'wave_example'
TRANSCIPTIONS_PATH = ROOT_PATH / 'data' / 'doc_example'
MY_WAV = ROOT_PATH / 'data' / 'moje_example'

MODEL_PATH = TMP_PATH / 'model.pkl'
RULES_PATH = TMP_PATH / 'rules.json'
MATRIX_PATH = TMP_PATH / 'confusion_matrix.json'
CONF_TABLE_PATH = TMP_PATH / 'confusion_table.json'
PHONETIC_SIMPLIFICATION_TABLE = ROOT_PATH / 'src' / 'simplification_table.tsv'


JULIUS_PATH = ROOT_PATH / 'julius'
JULIUS_EXE_PATH = JULIUS_PATH / 'julius-dnn.exe'


FFMPEG_PATH = ROOT_PATH / "ffmpeg" / "ffmpeg.exe"
FFPROBE_PATH = ROOT_PATH / "ffmpeg" / "ffprobe.exe"

