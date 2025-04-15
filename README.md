# Whisper experiments

## Setup

### 0. Check your python version (`python --version`), versions more recent than 3.12 are problematic with e.g. openai-whisper. `pyenv` or `conda` can help you select the python version
### 1. Setup and activate a virtualenv `python -m venv venv && . ./venv/bin/activate`
### 2. Install all the requirements

#### With GPU

```
pip install -r requirements.txt
```
#### Without GPU

```
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu torchaudio==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

## Running

### 1. Transcription

To transcribe all files run (use appropriate values for DATA_PATH):
```
find DATA_PATH -name '*.wav' -exec python transcribe.py {} +
```

This will transcribe all WAVE file using FasterWhisper and Stable-TS using the OpenAI Whisper and KB-Whisper models both in small and large version
All models required for the transcription are automatically downloaded.

### 2. Evaluation

To evaluate the transcription run (use appropriate values for GOLD_TRANSCRIPTION ,TRANSCRIPTION and OUTPUT):
```
python compare.py GOLD_TRANSCRIPTION TRANSCRIPTION > OUTPUT.tsv
``

The output is a table containig input filename, gold transcription, transcription, BLEU score, GLEU score and word error rate


## Common problems

- 
```
Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

Use `LD_PRELOAD=./venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9 python transcribe.py`

- `ImportError: libctranslate2-bc15bf3f.so.4.5.0: cannot enable executable stack as shared object requires: Invalid argument`

Run `patchelf --clear-execstack ./venv/lib/python3.12/site-packages/ctranslate2.libs/libctranslate2-bc15bf3f.so.4.5.0`
