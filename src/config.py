import pathlib


ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
TMP_PATH = ROOT_PATH / 'tmp'


JULIUS_PATH = ROOT_PATH / 'julius'
JULIUS_EXE_PATH = JULIUS_PATH / 'julius-dnn.exe'


FFMPEG_PATH = ROOT_PATH / "ffmpeg" / "ffmpeg.exe"
FFPROBE_PATH = ROOT_PATH / "ffmpeg" / "ffprobe.exe"

