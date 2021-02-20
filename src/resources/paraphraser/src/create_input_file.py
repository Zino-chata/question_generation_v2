import os
import sentencepiece as spm
import argparse
import sys
import json
lang='en'

sp = spm.SentencePieceProcessor()
#sp.Load(os.environ['MODEL_DIR'] + '/spm.model')
BASE_DIR='src/resources/paraphraser/'
sp.Load(BASE_DIR+'m39v1/spm.model')

def create_inputs():

    with open('generated_questions.json') as infile:
        data = json.load(infile)

    data_keys = list(data.keys())
    to_paraphrase=[]
    for key in data.keys():
        to_paraphrase.append(data[key]["question"])
        to_paraphrase.append(data[key]["question_variant"])


    sp_sents = [' '.join(sp.EncodeAsPieces(sent)) for sent in to_paraphrase]

    with open('test.src', 'wt') as fout:
        for sent in sp_sents:
            fout.write(sent + '\n')

    # we also need a dummy output file with the language tag
    with open('test.tgt', 'wt') as fout:
        for sent in sp_sents:
            fout.write(f'<{lang}> \n')


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--infile', type=str)
    #args = parser.parse_args()
    create_inputs()
