# ASR Post-Processing Error-Correction System

# Installation

## FFMPEG

Create directory with ffmpeg.exe and ffprobe.exe files downloaded from 
[FFMPEG site](https://www.ffmpeg.org/download.html). Change paths in 
`src/config.py` to point at correct paths.

## Julius

Julius ASR from 
[sourceforge](https://sourceforge.net/projects/juliusmodels/files/) is 
needed to run the project. Code was tested with PLPL-v7.1.Dnn.Bin.zip 
version of Julius. Change paths in `src/config.py` to point at correct 
paths.

## Python

Code was tested on python3.8. In order to install all required 
libraries run
```
pip3 install -r requirements.txt
```

## Data

As dataset is non public you will need your own data. Data needs to be 
organised into 3 directories:
- `WAV_ORIGINAL_PATH` containing WAVE recordings
- `TRANSCIPTIONS_PATH` containing transcriptions in DOC format. Each 
transcription needs to start with word "MR", every word before it will 
be discarded. This operation prevents processing identification codes 
present in transcriptions.
- `MY_WAV` containing WAVE recodings with the same names as 
`WAV_ORIGINAL_PATH`. It can be used to comapre results on alternative 
recordings assignmed to the same DOC file.

Change paths in `src/config.py` to point at correct paths.

# Usage

Run
``` sh
python3 src/main.py --help
```
for description of script parameters.

In order to run the program with default parameters run
``` sh
python3 src/main.py
```

## Log processing and visualisation

Outside of logs on the console script `src/main.py` creates directory 
`logs\experiment_{i}` with `i` being unique number for each execution 
(unless `--skip` option is used) and store all logs in there. Inside 
`src/visualisation_utils` there are 2 scripts for log processing, try 
them by running
```sh
python3 .\src\visualisation_utils\process_logs.py --input experiment_{i}
python3 .\src\visualisation_utils\process_time_logs.py --input experiment_{i}
```
Scripts will create directory `plots` with plots generated from supplied 
log folder.

## Similarity matrix

Post-processing method used by script `src/main.py` by default doesn't 
use similarity matrix. In order to change method you need to modify file 
`src/ASR_main_flow.py` so that the only uncommented line importing 
method is line
```
import methods.method_advanced13_matrix as method
```

Additionally before running main script you need to generate similarity 
matrix by running
```
python3 src/conf_table_creator.py
```
As a bonus this script also adds plots connected to similarity matrix to 
`plots` directory.

## Rule-based corrections

In order to used rule-based corrections before actual post-processing 
method you need to generate rules. In order to do that run
```
python3 src/ref_hyp_similarity.py
```
It will be used in subsequent runs of `src/main.py` script out of the 
box. As a bonus this script also adds plots connected to rules 
generation to `plots` directory is run with `--plot` argument.

## Caching

Every time N-gram model is generated it is saved in `tmp` directory. In 
order to use this model instead of training it each time add `--model` 
argument to `src/main.py` script
```
python3 src/main.py --model
```

Every time Julius prediction is generated it is saved in `tmp` 
directory. Unless deleted it will be used by default instead of calling 
Julius in subsequent runs of main script. Time of generating Julius 
predition is not included in `time.log` logs.
