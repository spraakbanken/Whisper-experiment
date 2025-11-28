from pathlib import Path

tmp_path = Path("tmp")
gold_path = Path(tmp_path / "gold")
data_path = Path(tmp_path / "Whisper-2025-Study")
transcription_path = Path(tmp_path / "transcription")
# reading_participants = ["F-32WJ-FB3B-S64F","F-3RQD-KCAQ-UGHK", "F-3WLV-CYG6-GXAL", "F-6W42-V2GN-MXNM", "F-79PA-NFUF-3HEQ", "F-AVF7-4HU9-N99R", "F-M9S8-UYGU-Q865", "F-MNFG-MFWA-H3KE", "F-NU94-Z6MG-K85T", "F-TXLH-R239-CPQG", "F-XTQN-WA4V-39Y7", "M-338H-F93S-5Y5P", "M-3TUP-R5PQ-D4WT", "M-84QY-JTL9-4H9C", "M-8AMP-SKYB-84H5", "M-8U3R-J54J-EVXC", "M-KE3V-PF7F-F59Y", "M-PWPP-D5EX-EIDN", "M-TK5R-Y7L4-458U", "M-W4N3-62QL-KMZN", "M-XPEU-R5UC-JYUJ"]
reading_participants = []
# cookietheft_participants = ["F-27KD-WDB8-YP2F","F-3N4A-3G7P-LHWE","F-5DJF-CPGZ-BUGQ","F-S8S8-TGDH-LG9H","F-UGLP-E76L-2F98","M-2MDJ-RVRY-UJPT","M-53AC-AP7Q-7UKA","M-LABM-75GG-5KH4","M-V56M-J6RZ-MCLU","MXPEU-R5UC-JYUJ"]
cookietheft_participants = ["F-27KD-WDB8-YP2F"]
semanticfluenct_participants = [ ]

tasks = ["CookieTheft", "Reading", "Semantic fluency"]
task_wav_files = {task: sorted(Path(data_path / task).glob("**/*.wav")) for task in tasks}
task_file_stems = {task: [transcription_path / task / f.stem for f in task_wav_files[task]] for task in tasks}
cuda_device = "cpu"
model_types = ["stable_ts", "openai"]
task_transcription_files = {task: [[str(f) + "_whisper_" + model_type + "_small_sv_0.5.txt" for f in task_file_stems[task]] for model_type in model_types] for task in tasks}

rule help:
    run:
        print("Run first rule `prepare` before executing `transcribe` and `evaluate`"),

# rule all:
#     input: # [task_transcript_files[task] for task in tasks]
#         task_transcription_files["Semantic fluency"] # + [f"{data_path}/Reading-transcriptions/{participant}-audio4-text2b.txt" for participant in reading_participants]

# rule transcribe:
#     params:
#         temperature="0.5",
#         device = cuda_device
#     script:
#         "transcribe.py"
        
# Extract the data from the zip file
rule extract:
    params:
        tmp_path=tmp_path
    input: f"{tmp_path}/Whisper-2025-Study.zip"
    output: directory(f"{tmp_path}/Whisper-2025-Study")
    log: "extract.log"
    shell: "unzip {input} -d {params.tmp_path}"

# Create the required directories for transcriptions and gold data
rule make_transcription_dir:
    input:
        rules.extract.output
#    output:
#        directory(f"{tmp_path}/whisper")
    shell:
        f"mkdir {transcription_path}"

# Create gold data dir and prepare the data
rule make_gold_data_dir:
    input:
        rules.extract.output
    output:
        [directory(f"{tmp_path}/gold")] +
        [f"{tmp_path}/gold/Reading/{participant}-audio4-text2b.txt" for participant in reading_participants]
    shell:
        "\n".join(
            # Make directory for gold data and each task
            [f"mkdir -p '{tmp_path}/gold/'"] +
            [f"mkdir -p '{tmp_path}/gold/{task}'" for task in tasks] +
            # Copy reading task gold data into individual file for each participant
            [ f"cp '{data_path}/Reading-transcriptions/Text 2b-transcription.txt' '{tmp_path}/gold/Reading/{participant}-audio4-text2b.txt'" for participant in reading_participants] +
            # Copy and fix cookie theft gold data
            [
                f"for f in '{data_path}/CookieTheft-transcriptions/'*.txt; do tail -n +2 $f > '{tmp_path}/gold/CookieTheft/'$(basename $f); done",
                f"mv '{tmp_path}/gold/CookieTheft/MXPEU-R5UC-JYUJ-audio2-kakburk.txt' '{tmp_path}/gold/CookieTheft/M-XPEU-R5UC-JYUJ-audio2-kakburk.txt'",
            ] +
            # Copy semantic fluency gold data
            [f"cp '{data_path}/Semantic fluency transcriptions/'*.txt '{tmp_path}/gold/Semantic fluency'"]
            )

rule prepare:
    input:
        rules.extract.output,
        rules.make_transcription_dir.output,
        rules.make_gold_data_dir.output
    run:
        print("You can execute rule `transcribe` now")

for task in tasks:
    for model_type in model_types:
        task_name = task.replace(' ', '_')
        rule:
            name: f"transcribe_{task_name}_{model_type}"
            params:
                device="cpu",
                model_type=model_type,
                language="sv",
                model_size="small",
                temperature="0.5",
                task_files=task_wav_files[task],
                transcription_path=transcription_path / task
            input:
                # These are only here to define dependencies and will be ignored by the script
                rules.make_gold_data_dir.output,
                rules.make_transcription_dir.output,
            log:
                f"transcribe_{task_name}_{model_type}.log"
            output:
                [str(f) + "_whisper_" + model_type + "_small_sv_0.5.txt" for f in task_file_stems[task]]
            script:
                "transcribe.py"


rule transcribe:
     input: [rules._rules[r].rule.output for r in rules._rules.keys() if rules._rules[r].rule.name in [f"transcribe_{task_name}_{model_type}" for task_name in [t.replace(' ','_') for t in tasks] for model_type in model_types]]
#         rules.transcribe_Semantic_fluency_stable_ts.output
#    input: [rules.__getattr__(transcription_rule).output for transcription_rule in [f"transcribe_{task_name}_{model_type}" for task_name in [t.replace(' ','_') for t in tasks] for model_type in model_types]]

rule clean:
    shell:
        "\n".join([
            f"rm -Rfv {gold_path}",
            f"rm -Rfv {transcription_path}",
            f"rm -Rfv {data_path}"
        ])
