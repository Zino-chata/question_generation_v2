import logging
import subprocess
import time

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.pipeline import ExtractiveQAPipeline

def qa_pipeline():
    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # ## Preprocessing of documents
    # Let's first fetch 517 Wikipedia articles for Game of Thrones some documents that we want to query
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    #write the docs to our DB.
    document_store.write_documents(dicts)

    # ## Initalize Retriever, Reader,  & Finder
    # Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered.
    retriever = ElasticsearchRetriever(document_store=document_store)

    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers.
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    #prediction = pipe.run(query="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
    prediction = pipe.run(query="'''Arya Stark''' portrayed by Maisie Williams.", top_k_retriever=10, top_k_reader=5)
    print_answers(prediction, details="medium")

if __name__ == "__main__":
    qa_pipeline()
