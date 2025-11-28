import sys
import os
from faster_whisper import WhisperModel
import stable_whisper
from pathlib import Path
import time
import csv

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Fixed parameters
condition_on_previous = True
languages = ['sv', '']
temperatures = [0,0.25,0.5,0.75,1]

if len(sys.argv) < 5:
    logger.error("Missing command line parameters, expected at least 4, got %d: %s", len(sys.argv)-1, sys.argv[1:])
    sys.exit(-1)
# Command-line parameters
model_name = sys.argv[1]
model_size = sys.argv[2]
device = sys.argv[3]
logger.info("Parameters model name: %s, model size: %s, device: %s", model_name, model_size, device)

model_parameters = {
        "device" : device,
}
if device == "cuda":
    model_parameters["compute_type"] = "float16"
else:
    model_parameters["compute_type"] = "float32"

logger.info("Loading model")
if model_name == "stable_ts":
    model = stable_whisper.load_faster_whisper(
        model_size,
        **model_parameters
    )
elif model_name == "openai":
    model = WhisperModel(
        model_size,
        **model_parameters
    )
elif model_name == "kb-whisper":
    model = WhisperModel(
        "KBLab/kb-whisper-" + model_size,
        **model_parameters
    )
else:
    logger.error("Unknown model name %s, use one of: stable_ts, openai, kb-whisper", model_name)
    sys.exit(-1)


audio_files = sys.argv[4:]

all_results = []
for audio_file in audio_files:
    logger.info("Transcribing %s", audio_file)
    for language in languages:
        for temperature in temperatures:
            logger.info("Setting temperature to %f", temperature)
            kwargs = {"temperature": temperature}
            if language:
                kwargs["language"] = language
                final_language = language
            if model_name == "stable_ts":
                start = time.process_time()
                result = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                end = time.process_time()
                if not language:
                    logger.info("Detected Language: " + result.language)
                    final_language = "auto:" + result.language
                text = result.text
                segments = result.segments
            else:
                start = time.process_time()
                segments, info = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                end = time.process_time()
                segments = list(segments)
                if not language:
                    logger.info("Detected Language: " + info.language)
                    final_language = "auto:" + info.language
                text = ' '.join([segment.text for segment in segments])
            logger.info("Took %ds to transcribe", end-start)
            all_results.append({'audio_file': Path(audio_file).stem, 'temperature': temperature, 'language': final_language, 'start': round(start,2), 'end': round(end,2), 'duration': round(end-start,2), 'text': text, 'segments': [{'start': round(float(s.start),2), 'end': round(float(s.end),2), 'text': s.text} for s in segments]})

with open('results_' + model_name + '_' + model_size +'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(all_results[0].keys())
    writer.writerows([r.values() for r in all_results])
