import sys
import os
from faster_whisper import WhisperModel
import stable_whisper
from pathlib import Path
import time

languages = ['sv', '']
condition_on_previous = True
model_sizes = [ "small", "large" ]
audio_files = sys.argv[1:]

models = {}
models["stable_ts"] = {}
models["openai"] = {}
models["kb_whisper"] = {}
for model_size in model_sizes:
    models["stable_ts"][model_size] = stable_whisper.load_faster_whisper(model_size)

    models["openai"][model_size] = WhisperModel(
        model_size,
        # device="cuda",
        device = "cpu",
        # compute_type="float16",
        compute_type="float32",
        download_root="cache", # cache directory
        # condition_on_previous_text = False # Can reduce hallucinations if we don't use prompts
    )
    models["kb_whisper"][model_size] = WhisperModel(
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
        for model_size in model_sizes:
            for language in languages:
                kwargs = {}
                if language:
                    kwargs["language"] = language
                    file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size, language])
                else:
                    file_name = "_".join([os.path.splitext(audio_file)[0], "whisper", m, model_size])
                text_file_name = file_name + ".txt"
                segments_file_name = file_name + ".segments"
                if not Path(text_file_name).exists():
                    print(text_file_name)
                    if m == "stable_ts":
                        start = time.process_time()
                        result = models[m][model_size].transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                        end = time.process_time()
                        print("Took " +str(end-start) + " s to transcribe " + text_file_name)
                        text = result.text
                    else:
                        start = time.process_time()
                        segments, info = models[m][model_size].transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                        end = time.process_time()
                        print("Took " +str(end-start) + " s to transcribe " + text_file_name)
                        print("Detected Language: " + info.language)
                        text = ' '.join([segment.text for segment in segments])
                    with open(text_file_name, 'w') as f:
                        f.write(text)
                if not Path(segments_file_name).exists():
                    print(segments_file_name)
                    lines = []
                    index = 1
                    if m == "stable_ts":
                        start = time.process_time()
                        result = models[m][model_size].transcribe(audio_file, condition_on_previous_text=condition_on_previous, **kwargs)
                        end = time.process_time()
                        print("Took " +str(end-start) + " s to transcribe " + segments_file_name)
                        for segment in result.segments:
                            lines.append("\t".join([str(index),str(segment.start),str(segment.end),segment.text]))
                            index+=1
                    else:
                        start = time.process_time()
                        segments, info = models[m][model_size].transcribe(audio_file, condition_on_previous_text=condition_on_previous, log_progress=True, **kwargs)
                        end = time.process_time()
                        print("Took " +str(end-start) + " s to transcribe " + segments_file_name)
                        for segment in segments:
                            lines.append("\t".join([str(index),str(segment.start),str(segment.end),segment.text]))
                            index+=1

#                        text = ' '.join([segment.text for segment in segments])
                    with open(segments_file_name, 'w') as f:
                        text = "\n".join(lines)
                        f.write(text)
# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# #print(segments)
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
