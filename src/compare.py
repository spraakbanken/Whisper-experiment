import os
import os.path
import sys
import glob
import re
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
import werpy
import pprint
import csv
import logging

tokenizer = RegexpTokenizer(r'\w+')

chencherry = SmoothingFunction()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

gold_data = dict()
data = []
if __name__=="__main__":
  if len(sys.argv) > 3:
    output_file = sys.argv[1]
    gold_file = sys.argv[2]
    transcription_files = sys.argv[3:]
  else:
    print(f"Usage: %s output_csv gold_csv transcription_csv" % sys.argv[0])
    exit(0)

  participant_pattern = re.compile("([FM]-)?([A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4})")
  model_pattern = re.compile("results_([\\w-]+)_(\\w+)")
  with open(gold_file, newline='') as csvfile:
    logger.info("Reading gold data from %s", gold_file)
    gold_reader = csv.DictReader(csvfile)
    for gold_dataset in gold_reader:
      participant = participant_pattern.match(gold_dataset['filename']).group(2)
      gold_data[participant]=dict()
      gold_data[participant]['gold_file']=gold_dataset['filename']
      gold_data[participant]['gold_text']=gold_dataset['text']
  for transcription_file in transcription_files:
    with open(transcription_file, newline='') as csvfile:
      logger.info("Reading transcription from %s", transcription_file)
      transcription_reader = csv.DictReader(csvfile)
      if 'model_name' not in transcription_reader.fieldnames:

        model_name, model_size = model_pattern.match(os.path.basename(transcription_file)).groups()
      for transcription_dataset in transcription_reader:
        participant = participant_pattern.match(os.path.basename(transcription_dataset['audio_file'])).group(2)
        if 'model_name' in transcription_dataset:
          model_name = transcription_dataset['model_name']
          model_size = transcription_dataset['model_size']
        gold_text = werpy.normalize(gold_data[participant]['gold_text'])
        transcription_text = werpy.normalize(transcription_dataset['text'])
        gold_set=set(gold_text.split())
        transcription_set=set(transcription_text.split())
        tp = len(transcription_set.intersection(gold_set))
        tn = 0 # TODO ?
        fp = len(transcription_set.difference(gold_set))
        fn = len(gold_set.difference(transcription_set))
        data.append({
          'participant': participant,
          'gold_file': gold_data[participant]['gold_file'], 'gold_text': gold_data[participant]['gold_text'], 'gold_text_normalized': gold_text,
          'audio_file': transcription_dataset['audio_file'],
          'model_name': model_name, 'model_size': model_size, 'temperature': transcription_dataset['temperature'],
          'language': transcription_dataset['language'],
          'transcription_start_time': transcription_dataset['start'] ,'transcription_end_time': transcription_dataset['end'], 'transcription_duration': transcription_dataset['duration'],
          'transcription_text': transcription_dataset['text'], 'transcription_segments': transcription_dataset['segments'], 'transcription_text_normalized': transcription_text,
          'bleu_score': sentence_bleu([gold_text], transcription_text, smoothing_function=chencherry.method1),
          'gleu_score': sentence_gleu([gold_text], transcription_text),
          'word_error_rate': werpy.wer(gold_text, transcription_text),
          'precision': tp/(tp+fp),
          'recall': tp/(tp+fn),
          'accuracy': (tp+tn)/(tp+tn+fp+fn),
          'f1':(2*tp)/(2*tp+fp+fn)
        })

  with open(output_file, 'w', newline='') as f:
    logger.info("Write analysis to %s", output_file)
    writer = csv.writer(f)
    writer.writerow(data[0].keys())
    writer.writerows([d.values() for d in data])
  # gold_files = glob.glob(gold_path + "/*.txt")
  pprint.pp(data)
# gold_files = glob.glob(gold_path + "/*.txt")
# gold_files.sort()
# transcription_files = glob.glob(transcription_path + "/*.txt")
# transcription_files.sort()
# transcriptions = {}
# for g in gold_files:
#     (gpath, gfilename) = os.path.split(g)
#     (gname, _) = os.path.splitext(gfilename)
#     transcriptions[gname]={}
#     transcriptions[gname]["gold"] = tokenizer.tokenize(open(g).read().lower())
# for t in transcription_files:
#     (tpath, tfilename) = os.path.split(t)
#     (tname, _) = os.path.splitext(tfilename)
#     result = re.match("([\\w-]+?)_.+", tname)
#     if result is not None:
#       transcriptions[result[1]][tname] = tokenizer.tokenize(open(t).read().lower())
# texts = list(transcriptions.keys())
# texts.sort()
# print("Version\tTranscription\tGold\tBLEU\tGLEU\tWER")
# for text in texts:
#     gold = " ".join(transcriptions[text]["gold"])
#     versions = list(transcriptions[text].keys())
#     versions.remove("gold")
#     for v in versions:
#       transcription = " ".join(transcriptions[text][v])
#       bleu = 
#       gleu = 
#       wer = 
#       if wer is not None:
#         print(f"%s\t\"%s\"\t\"%s\"\t%f\t%f\t%f" % (v, transcription, gold, bleu, gleu, wer))
#       else:
#         sys.stderr.write(f"WER failed for GOLD \"%s\" and TRANSCRIPTION \"%s\"" % (gold, transcription))
# #        print(f"%s\t\"%s\"\t\"%s\"\t%f\t%f\t%f" % (v, transcription, gold, bleu, gleu, -1))
# #      else:
      
