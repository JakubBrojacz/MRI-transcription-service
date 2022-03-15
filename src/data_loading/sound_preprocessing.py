import os
import pathlib
from pydub import AudioSegment
import noisereduce
from scipy.io import wavfile

import config

os.environ["PATH"] += os.pathsep + str(config.FFMPEG_PATH.parent)

AudioSegment.converter = str(config.FFMPEG_PATH)
AudioSegment.ffmpeg = str(config.FFMPEG_PATH)
AudioSegment.ffprobe = str(config.FFPROBE_PATH)


def convert_m4a_wav(input_file: pathlib.Path, output_dir: pathlib.Path):
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / (input_file.stem + '.wav')
    if input_file.suffix == '.wav':
        track = AudioSegment.from_file(input_file, 'wav')
    else:
        track = AudioSegment.from_file(input_file, 'm4a')
    track = track.set_frame_rate(16000)
    track = track.set_channels(1)
    track += 30
    track = track.low_pass_filter(1000).high_pass_filter(1000)
    track += 10
    track.export(output_filename, 'wav', bitrate="192k")
    rate, data = wavfile.read(output_filename)
    # reduced_noise = noisereduce.reduce_noise(y=data, sr=rate)
    # output_filename1 = output_dir / (input_file.stem + '1.wav')
    # wavfile.write(output_filename1, rate, reduced_noise)
    # track = AudioSegment.from_file(output_filename1, 'wav')
    # track += 15
    # output_filename2 = output_dir / (input_file.stem + '2.wav')
    # track.export(output_filename2, 'wav', bitrate="192k")
    return output_filename


if __name__ == '__main__':
    convert_m4a_wav(pathlib.Path(
        r'C:\Users\kubab\OneDrive\Dokumenty\Nagrania dźwiękowe\Nagranie (5).m4a'), pathlib.Path('../tmp'))
