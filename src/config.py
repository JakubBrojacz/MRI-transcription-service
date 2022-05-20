import pathlib


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
TMP_PATH = ROOT_PATH / 'tmp'
LOG_PATH = ROOT_PATH / 'logs'
MODEL_PATH = TMP_PATH / 'model.pkl'
WAV_ORIGINAL_PATH = pathlib.Path(".\\data_conf\\mgr\\mimi\\wav_files")
TRANSCIPTIONS_PATH = pathlib.Path(".\\data_conf\\mgr\\mimi\\opisy")
MY_WAV = pathlib.Path(".\\data_conf\\mgr\\moje_nagrania")


JULIUS_PATH = ROOT_PATH / 'julius'
JULIUS_EXE_PATH = JULIUS_PATH / 'julius-dnn.exe'


FFMPEG_PATH = ROOT_PATH / "ffmpeg" / "ffmpeg.exe"
FFPROBE_PATH = ROOT_PATH / "ffmpeg" / "ffprobe.exe"

