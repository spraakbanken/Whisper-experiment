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

def transcribe(model_name : str, model_size: str, device : str, audio_files: list[str]):
    # Fixed parameters
    condition_on_previous = True
    #languages = ['sv']
    languages = ['sv']
    temperatures = [0,0.25,0.5,0.75,1]
    # temperatures = [0.5]
    prompt = ""
    # First attempt prompting trip to stockholm
#    prompt = "a aa ah aha ahja aja betal billigaste blidde booking brukade brukskar com dar denne ditut dom dä däruppe därute e ee eee eeee eh eja emilie er erbjuda erbjudan fjorton fotbollsförbundet framför fruängen gjord haha hahaha hihi hänger hållspänner höll ihåg inkvarteringen iordning jaa jae jobbade ka kalendarium kapong karlbergsvägen kriminaler kulturellt la lede liggandes liksom litegrann lugn lättaste lå låg långt lösa m medans meddelat menar metron mm mobil mtrx ne neccessär neccessären nuförtiden nutid ny nyöppnade nä nänä närmar näö nåra nåt o oh ok orkester pokemon pokemonfigurer punkt rosaexpress rundar runtom runtomkring ryggen sade senvåren sextiotre sidan sjöstan sliper so sofiahemmet soppor stadsdelarna stället sådan såpass tidn tillfällena transportsätt trosor tu veta vintersakerna väldig västtrafik väx yr äldste äro åk åkaruntbuss åu öe"
    # Second attempt prompting
    #prompt = "Glossary: a, aa, agoda, ah, aha, ahja, aja, alls, allt, alltså, att, avresedatum, bara, behöver, behövs, bestämma, betal, biljett, biljetter, billig, billigaste, blev, blidde, boendet, booking, bor, borta, brevlåda, brukade, brukskar, com, dar, dator, de, dem, den, denne, ditut, dom, du, dä, där, därför, däruppe, därute, då, e, ee, eee, eeee, eh, eja, emilie, er, erbjuda, erbjudan, ett, faktiskt, fixar, fjorton, fotbollsmatchen, framför, frun, fruängen, funderar, få, förbereda, förbereder, givetvis, gjord, grand, grannar, granne, grejer, gången, göra, göteborgsposten, ha, haha, hahaha, hand, heter, hihi, hittar, hotell, hur, husdjur, huset, här, hållspänner, höll, ihop, ihåg, internet, iordning, iväg, ja, jaa, jae, jo, jobbade, ju, ka, kalendarium, kanske, kapong, karlbergsvägen, klara, klart, kolla, kommit, kriminaler, kulturellt, kunde, la, lagt, lakan, lede, liksom, lite, litegrann, lugn, längesen, lättaste, lå, låg, långt, lösa, m, man, meddelat, men, menar, mera, metron, mina, mm, mobil, mtrx, många, möjligtvis, ne, neccessär, neccessären, nej, nja, nuförtiden, nutid, ny, nä, nähä, nänä, när, nära, närmar, nätter, näö, någon, någonstans, något, några, nåra, nåt, o, oh, ok, okej, om, ombyten, orkester, packa, packar, packarlista, pokemon, pokemonfigurer, post, punkt, retur, rosaexpress, rundar, runt, runtom, runtomkring, ryggen, sa, sade, schampo, se, sedan, senvåren, ser, sevärdheter, sextiotre, sidan, sjöstan, skall, sköta, sliper, so, sockar, sofiahemmet, soppor, staden, stadsdelarna, stockholms, stup, stället, säga, sådan, sådana, sådant, såpass, ta, tar, telefon, tidn, tidning, tillbaks, tillfällena, titta, transportsätt, tror, trosor, tröja, tu, tur, tycker, tänker, upp, utan, utav, va, valdemarsudde, vasa, vattna, vattnar, vet, veta, vid, vilja, vilken, vill, vintersakerna, väl, väldig, vänner, västtrafik, väx, yr, är, äro, åk, åkaruntbuss, årstid, åt, åu, öe"
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

    all_results = []
    for audio_file in audio_files:
        logger.info("Transcribing %s", audio_file)
        for language in languages:
            for temperature in temperatures:
                logger.info("Setting temperature to %f", temperature)
                kwargs = {"temperature": temperature}
                if prompt:
                    kwargs["initial_prompt"]=prompt
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
                all_results.append({'audio_file': Path(audio_file).stem, 'model_name': model_name, 'model_size': model_size, 'temperature': temperature, 'language': final_language, 'start': round(start,2), 'end': round(end,2), 'duration': round(end-start,2), 'text': text, 'segments': [{'start': round(float(s.start),2), 'end': round(float(s.end),2), 'text': s.text} for s in segments]})
    # Detroy model?
    del model
    return all_results

def save_csv(filename: str, all_results: list[dict]) -> None:
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(all_results[0].keys())
        writer.writerows([r.values() for r in all_results])

if __name__ == "__main__":
    if len(sys.argv) < 5:
        logger.error("Missing command line parameters, expected at least 4, got %d: %s", len(sys.argv)-1, sys.argv[1:])
        sys.exit(-1)
    model_name = sys.argv[1]
    model_size = sys.argv[2]
    device = sys.argv[3]
    audio_files = sys.argv[4:]
    logger.info("Parameters model name: %s, model size: %s, device: %s, audio files: %s", model_name, model_size, device, audio_files)
    output_file = 'results_' + model_name + '_' + model_size +'.csv'
    all_results = transcribe(model_name, model_size, device, audio_files)
    save_csv(output_file, all_results)
