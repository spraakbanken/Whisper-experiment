import sys
import os
from faster_whisper import WhisperModel
import stable_whisper
from pathlib import Path
import time

from snakemake.script import snakemake
import logging
if len(snakemake.log) >= 1:
    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(asctime)s %(filename)s %(message)s')
else:
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format='%(asctime)s %(filename)s %(message)s',)

logger = logging.getLogger(__name__)
condition_on_previous = True
language = snakemake.params["language"]
model_type = snakemake.params["model_type"]
model_size = snakemake.params["model_size"]
device = snakemake.params["device"]

if device == "cuda":
    compute_type = "float16"
else:
    compute_type="float32"

temperature = float(snakemake.params["temperature"])

audio_files = snakemake.params["task_files"]
transcription_files = snakemake.output
transcription_path = Path(snakemake.params["transcription_path"])

logging.info("Output: " + str(transcription_files))

model_parameters = {
        "device": device,
        "compute_type" : compute_type,
#        "download_root" : "cache"
        # condition_on_previous_text = False # Can reduce hallucinations if we don't use prompts
}

if model_type == "stable_ts":
    model = stable_whisper.load_faster_whisper(
        model_size,
        **model_parameters
    )
elif model_type == "openai":
    model = WhisperModel(
        model_size,
        **model_parameters
    )
elif model_type == "kb_whisper":
    WhisperModel(
        "KBLab/kb-whisper-" + model_size,
        **model_parameters
    )
else:
    logger.error("No such model defined: %s", model_type)
    sys.exit(-1)

for audio_file in audio_files:
    logger.info("Transcribing: %s",audio_file)
    basename = Path(audio_file).stem
    kwargs = {"temperature": temperature}
    if language:
        kwargs["language"] = language
        file_name = "_".join([str(transcription_path / basename), "whisper", model_type, model_size, language, str(temperature)])
    else:
        file_name =  "_".join([str(transcription_path / basename), "whisper", model_type, model_size, "auto", str(temperature)])
    text_file_name = file_name + ".txt"
    segments_file_name = file_name + ".segments"
    write_transcription = any([t.find(text_file_name) for t in transcription_files])
    write_segments = any([t.find(segments_file_name) for t in transcription_files])
    do_transcribe = any([t.find(basename) for t in transcription_files])
    logger.info("%s %s %s",str(write_transcription), str(write_segments), str(do_transcribe))
    # if do_transcribe:
    #     start = time.process_time()
    #     if model_type == "stable_ts":
    #         result = model.transcribe(str(audio_file), condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
    #         text = result.text
    #     else:
    #         segments, info = model.transcribe(str(audio_file), condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
    #         logger.debug("Detected Language: %s", info.language)
    #         text = ' '.join([segment.text for segment in segments])
    #     end = time.process_time()
    #     logger.info("Took %ss to transcribe %s",end-start, text_file_name)
    #     if write_transcription:
    #         logger.debug("Creating transcription file: %s", text_file_name)
    #         with open(text_file_name, 'w') as f:
    #             f.write(text)
    #     if write_segments:
    #         logger.debug("Creating transcription file: %s", segments_file_name)
    #         lines = []
    #         index = 1
    #         if model_type == "stable_ts":
    #             segments = result.segments
    #         for segment in segments:
    #             lines.append("\t".join([str(index),str(segment.start),str(segment.end),segment.text]))
    #             index+=1
    #         with open(segments_file_name, 'w') as f:
    #             text = "\n".join(lines)
    #             f.write(text)
