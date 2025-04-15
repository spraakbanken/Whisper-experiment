import os
import os.path
import sys
import glob
import re
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
import werpy

tokenizer = RegexpTokenizer(r'\w+')

chencherry = SmoothingFunction()

if len(sys.argv) > 2:
  gold_path = sys.argv[1]
  transcription_path = sys.argv[2]
else:
  print(f"Usage: %s gold_path transcription_path" % sys.argv[0])
  exit(0)
gold_files = glob.glob(gold_path + "/*.txt")
gold_files.sort()
transcription_files = glob.glob(transcription_path + "/*.txt")
transcription_files.sort()
transcriptions = {}
for g in gold_files:
    (gpath, gfilename) = os.path.split(g)
    (gname, _) = os.path.splitext(gfilename)
    transcriptions[gname]={}
    transcriptions[gname]["gold"] = tokenizer.tokenize(open(g).read().lower())
for t in transcription_files:
    (tpath, tfilename) = os.path.split(t)
    (tname, _) = os.path.splitext(tfilename)
    result = re.match("([\\w-]+?)_.+", tname)
    if result is not None:
      transcriptions[result[1]][tname] = tokenizer.tokenize(open(t).read().lower())
texts = list(transcriptions.keys())
texts.sort()
print("Version\tTranscription\tGold\tBLEU\tGLEU\tWER")
for text in texts:
    gold = " ".join(transcriptions[text]["gold"])
    versions = list(transcriptions[text].keys())
    versions.remove("gold")
    for v in versions:
      transcription = " ".join(transcriptions[text][v])
      bleu = sentence_bleu([transcriptions[text]["gold"]],transcriptions[text][v],smoothing_function=chencherry.method1)
      gleu = sentence_gleu([transcriptions[text]["gold"]],transcriptions[text][v])
      wer = werpy.wer(gold, transcription)
      if wer is not None:
        print(f"%s\t\"%s\"\t\"%s\"\t%f\t%f\t%f" % (v, transcription, gold, bleu, gleu, wer))
      else:
        sys.stderr.write(f"WER failed for GOLD \"%s\" and TRANSCRIPTION \"%s\"" % (gold, transcription))
#        print(f"%s\t\"%s\"\t\"%s\"\t%f\t%f\t%f" % (v, transcription, gold, bleu, gleu, -1))
#      else:
      
