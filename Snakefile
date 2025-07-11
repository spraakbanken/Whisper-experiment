from pathlib import Path

tmp_path = Path("tmp")
participants = ["F-32WJ-FB3B-S64F","F-3RQD-KCAQ-UGHK", "F-3WLV-CYG6-GXAL", "F-6W42-V2GN-MXNM", "F-79PA-NFUF-3HEQ", "F-AVF7-4HU9-N99R", "F-M9S8-UYGU-Q865", "F-MNFG-MFWA-H3KE", "F-NU94-Z6MG-K85T", "F-TXLH-R239-CPQG", "F-XTQN-WA4V-39Y7", "M-338H-F93S-5Y5P", "M-3TUP-R5PQ-D4WT", "M-84QY-JTL9-4H9C", "M-8AMP-SKYB-84H5", "M-8U3R-J54J-EVXC", "M-KE3V-PF7F-F59Y", "M-PWPP-D5EX-EIDN", "M-TK5R-Y7L4-458U", "M-W4N3-62QL-KMZN", "M-XPEU-R5UC-JYUJ"]

male = [p for p in participants if p.startswith("M")]
female = [p for p in participants if p.startswith("F")]

path = Path(tmp_path / "Whisper-2025-Study")
tasks = ["CookieTheft", "Reading", "Semantic fluency"]
task_wav_files = {task: sorted(Path(path / task).glob("*.wav")) for task in tasks}
task_file_stems = {task: [f.parent / f.stem for f in task_wav_files[task]] for task in tasks}
cuda_device = "cpu"
model_types = ["stable_ts", "openai"]
task_transcription_files = {task: [[str(f) + "_whisper_" + model_type + "_small_sv_0.5.txt" for f in task_file_stems[task]] for model_type in model_types] for task in tasks}

rule all:
    input: # [task_transcript_files[task] for task in tasks]
        task_transcription_files["Semantic fluency"]

rule transcribe:
    params:
        temperature="0.5",
        device = cuda_device
    script:
        "transcribe.py"
        
for task in tasks:
    for model_type in model_types:
        rule:
            name: f"transcribe_{task}_{model_type}"
            params:
                device="cpu",
                model_type=model_type,
                language="sv",
                model_size="small",
                temperature="0.5"
            input:
                task_wav_files[task]
            log:
                f"transcribe_{task}_{model_type}.log"
            output:
                [str(f) + "_whisper_" + model_type + "_small_sv_0.5.txt" for f in task_file_stems[task]]
            script:
                "transcribe.py"
         
rule extract:
    params:
        tmp_path=tmp_path
    input: expand("{tmp_path}/Whisper-2025-Study.zip", tmp_path=tmp_path)
    output: directory(expand("{tmp_path}/Whisper-2025-Study", tmp_path=tmp_path))
    log: "extract.log"
    shell: "unzip {input} -d {params.tmp_path}"
