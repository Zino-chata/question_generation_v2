#!/usr/bin/env bash

#download model for the first time otherwise export only
MODEL_DIR="m39v1"
if [ -d $MODEL_DIR ]; then
  export MODEL_DIR=m39v1/
else
  echo "Downloading model"
  wget http://data.statmt.org/prism/m39v1.tar
  tar xf m39v1.tar
  export MODEL_DIR=m39v1/
fi;

#create test bin with sents to be paraphrased
:'
fairseq-preprocess --source-lang src --target-lang tgt  \
    --joined-dictionary  --srcdict $MODEL_DIR/dict.tgt.txt \
    --trainpref  test  --validpref test  --testpref test --destdir test_bin

#generate paraphrases
python -W ignore -u src/generate_paraphrases.py test_bin --batch-size 8 \
   --prefix-size 1 \
   --path $MODEL_DIR/checkpoint.pt \
   --prism_a 0.003 --prism_b 4
'