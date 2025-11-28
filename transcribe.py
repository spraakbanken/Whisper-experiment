import sys
import os
from faster_whisper import WhisperModel
import stable_whisper
from pathlib import Path
import time

# Fixed parameters
condition_on_previous = True
languages = ['sv', '']
temperatures = [0,0.25,0.5,0.75,1]

# Command-line parameters
model_name = sys.argv[1]
model_size = sys.argv[2]
device = sys.argv[3]
logger.info("Parameters model name: %s, model size: %s, device: ", model_name, model_size, device)

model_parameters = {
        "device" : device,
}
if device == "cuda":
    model_parameters["compute_type"] = "float32"
else:
    model_parameters["compute_type"] = "float16"

logger.info("Loading model")
if model_name == "stable_ts":
    model = stable_whisper.load_faster_whisper(
        model_size,
        **model_parameters
    )
elif model_name == "open_ai":
    model = WhisperModel(
        model_size,
        **model_parameters
    )
elif model_name == "kb_whisper"
    model = WhisperModel(
        "KBLab/kb-whisper-" + model_size,
        **model_parameters
    )
else:
    logger.error("Unknown model name %s", model_name)
    system.exit(-1)


audio_files = sys.argv[4:]

for audio_file in audio_files:
    logger.info("Transcribing %s", audio_file)
    for language in languages:
        for temperature in temperatures:
            kwargs = {"temperature": temperature}
            if language:
                kwargs["language"] = language
                file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size, language, str(temperature)])
            else:
                file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size, "auto", str(temperature)])
            text_file_name = file_name + ".txt"
            segments_file_name = file_name + ".segments"
            if not Path(text_file_name).exists():
                if model_name == "stable_ts":
                    start = time.process_time()
                    result = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                    end = time.process_time()
                    text = result.text
                else:
                    start = time.process_time()
                    segments, info = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                    end = time.process_time()
                    if not language:
                        logger.info("Detected Language: " + info.language)
                    text = ' '.join([segment.text for segment in segments])
                logger.info("Took %ds to transcribe", end-start)
                logger.info("Writing transcription %s", text_file_name)
                with open(text_file_name, 'w') as f:
                    f.write(text)
                if not Path(segments_file_name).exists():
                    lines = []
                    index = 1
                    if model_name == "stable_ts":
                        start = time.process_time()
                        result = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                        end = time.process_time()
                        for segment in result.segments:
                            lines.append("\t".join([str(index),str(segment.start),str(segment.end),segment.text]))
                            index+=1
                    else:
                        start = time.process_time()
                        segments, info = model.transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                        end = time.process_time()
                        for segment in segments:
                            lines.append("\t".join([str(index),str(segment.start),str(segment.end),segment.text]))
                            index+=1
                    logger.info("Took %ds to transcribe", end-start)
                    logger.info("Writing segments %s", segments_file_name)
                    with open(segments_file_name, 'w') as f:
                        text = "\n".join(lines)
                        f.write(text)
