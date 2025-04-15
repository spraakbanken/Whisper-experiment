import sys
import os
from faster_whisper import WhisperModel
import stable_whisper
from pathlib import Path

languages = ['sv', '']
condition_on_previous = True
# model_size = "small"
model_size = "large"
audio_files = sys.argv[1:]
models = {}
models["stable_ts"] = stable_whisper.load_faster_whisper(model_size)

models["openai"] = WhisperModel(
    model_size,
    # device="cuda",
    device = "cpu",
    # compute_type="float16",
    compute_type="float32",
    download_root="cache", # cache directory
    # condition_on_previous_text = False # Can reduce hallucinations if we don't use prompts
)
models["kb_whisper"] = WhisperModel(
    "KBLab/kb-whisper-" + model_size,
    # device="cuda",
    device = "cpu",
    # compute_type="float16",
    compute_type="float32",
    download_root="cache", # cache directory
    # condition_on_previous_text = False # Can reduce hallucinations if we don't use prompts
)
# if len(sys.argv) > 2:
#     generate_kwargs["language"] = sys.argv[2]
for audio_file in audio_files:
    print(audio_file)
    for m in models.keys():
        for language in languages:
            kwargs = {}
            if language:
                kwargs["language"] = language
                text_file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size, language]) + ".txt"
            else:
                text_file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size]) + ".txt"
            if not Path(text_file_name).exists():
                print(text_file_name)
                if m == "stable_ts":
                    result = models[m].transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                    text = result.text
                else:
                    segments, info = models[m].transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                    print("Detected Language: " + info.language)
                    text = ' '.join([segment.text for segment in segments])
                with open(text_file_name, 'w') as f:
                    f.write(text)
# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# #print(segments)
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
