import csv
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from pprint import pp
import string2string.alignment
import werpy
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'(\w+|[^\w\s]+)')

def tonkenize1(text):
    return werpy.normalize(text).split()
def tokenize2(text):
    return tokenizer.tokenize(text.lower())
def align1(text1,text2):
    # Create sequences to be aligned.
    a = Sequence(text1)
    b = Sequence(text2)

    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    aEncoded = v.encodeSequence(a)
    bEncoded = v.encodeSequence(b)
    
    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
    
    # Iterate over optimal alignments and print them.
    for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        print(alignment)
        print('Alignment score:', alignment.score)
        print('Percent identity:', alignment.percentIdentity())
def align2(text1,text2):
    aligner = string2string.alignment.NeedlemanWunsch()
    result = aligner.get_alignment(text1,text2,return_score_matrix=True)
    print(result[0])
    print(result[1])
    print(max([max(l) for l in result[2]]))
with open('results_kb-whisper_large_multiple_speakers.csv') as c:
    text_uncompressed=[r['text'] for r in csv.DictReader(c)]
with open('results_kb-whisper_large_multiple_speakers_compressed.csv') as c:
    text_compressed=[r['text'] for r in csv.DictReader(c)]
#for i in range(0,1): # len(text_uncompressed)):
for i in range(0,len(text_uncompressed),5):
    align2(tokenize2(text_uncompressed[i]), tokenize2(text_compressed[i]))