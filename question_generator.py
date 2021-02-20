import logging
import subprocess
import time
import sys
import pprint
import re
import random
import secrets
import numpy as np
import json
import subprocess
import shlex

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_file_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

################TF_DF############################
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import TfidfRetriever

####TRANSLATOR#########################
from haystack.translator.transformers import TransformersTranslator

####SUMMARIZER#########################
from haystack.summarizer.transformers import TransformersSummarizer
from haystack import Document

def qa_pipeline():

    #document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store = InMemoryDocumentStore()

    doc_dir = "data/document"

    # convert files to dicts containing documents that can be indexed and perform NER
    dicts, ner, faq, qg = convert_file_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text) #zino: convert_files_to_dicts -> convert_file_to_dicts

    #write the docs to DB.
    document_store.write_documents(dicts)

    #Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered.
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Paraphrase questions by back-translation
    translator_fr = TransformersTranslator("Helsinki-NLP/opus-mt-en-fr")
    translator_en = TransformersTranslator("Helsinki-NLP/opus-mt-fr-en")

    #summarize given answers
    summarizer=TransformersSummarizer("google/pegasus-xsum")

    generated_qas={}
    #generated_qas["qg_sents"]= qg_sent(retriever, reader, pipe, qg)
    #generated_qas["para_original_question"] = paraphraser(translator_fr, translator_en, faq)
    generated_qas["summ_orig_answer"] = summarize(summarizer, faq)

    #write to file
    generated_qas = json.dumps(generated_qas, indent=2)
    with open('outputs/temp_eqBank2.json', 'w') as outfile:
        outfile.write(generated_qas)


#method1: generate questions on sentence-level
def qg_sent(retriever, reader, pipe, qg):
    qa_pairs = []
    for question in qg:
        current_pair={}
        answer = retriever.retrieve_(query=question, top_k=3)
        current_pair["question"] = question
        current_pair["answer"] = ' '.join(list(set(answer)))

        prediction = pipe.run(query=question, top_k_retriever=3, top_k_reader=1)
        pred = prediction["answers"]
        if pred:  # not empty
            if pred[0]["score"] >= 1:
                current_pair["score"]=pred[0]["score"]
                qa_pairs.append(current_pair)

    print("Generated ", len(qa_pairs), " questions")

    return qa_pairs

#method2: paraphrase original questions through back-translation -->en-fr-en
def paraphraser(trans_fr, trans_en, qa):
    qa_pairs=[]
    for q_a in qa:
        current_pair={}
        current_pair["original_question"] = q_a["question_new"]
        paraphrase = trans_fr.translate(query=current_pair["original_question"])
        current_pair["question"] = trans_en.translate(query=paraphrase)
        #ensure they are different
        if current_pair["original_question"] != current_pair["question"].lower():
            current_pair["answer"] = q_a["answer"]
            qa_pairs.append(current_pair)

    print("Generated ", len(qa_pairs), " questions")
    return qa_pairs

#method3: summarize original answers
def summarize(summarizer_,qa):
    qa_pairs = []
    for q_a in qa:
        current_pair = {}
        current_pair["question"] = q_a["question_new"]
        to_summ = [Document(text=q_a["answer"])]
        summarized_answer = summarizer_.predict(to_summ,generate_single_summary=True)
        current_pair["answer"]=summarized_answer[0]["text"]
        print(current_pair["answer"])

        # ensure they are different
        if current_pair["answer"].lower() != q_a["answer"].lower():
            current_pair["original_answer"]=q_a["answer"]
            qa_pairs.append(current_pair)

    print("Generated ", len(qa_pairs), " questions")
    return qa_pairs

if __name__ == "__main__":
    qa_pipeline()
