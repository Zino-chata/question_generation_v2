#!/usr/bin/env bash

BASE_DIR=question_gen/resources/paraphraser
MODEL_DIR=$BASE_DIR/m39v1
export MODEL_DIR=$BASE_DIR/m39v1

#remove test_bin if it exists
if [ -d test_bin ]; then
  rm -Rf test_bin
fi

#create input file
python -W ignore -u question_gen/resources/paraphraser/src/create_input_file.py

#create test bin with sents to be paraphrased
fairseq-preprocess --source-lang src --target-lang tgt  \
    --joined-dictionary  --srcdict $MODEL_DIR/dict.tgt.txt \
    --trainpref  test  --validpref test  --testpref test --destdir test_bin

#generate paraphrases
python -W ignore -u question_gen/resources/paraphraser/src/generate_paraphrases.py test_bin --batch-size 8 \
   --prefix-size 1 \
   --path $MODEL_DIR/checkpoint.pt \
   --prism_a 0.003 --prism_b 4

