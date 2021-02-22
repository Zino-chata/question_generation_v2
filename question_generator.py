import time
start = time.time()

import logging
import sys
import pprint
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import os

from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_file_to_dicts, sent_tokenize
from haystack.utils import print_answers

################TF_DF############################
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.retriever.dense import EmbeddingRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

####TRANSLATOR#########################
from haystack.translator.transformers import TransformersTranslator

def qa_pipeline():

    #document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
    document_store = InMemoryDocumentStore()

    doc_dir = "data/document"

    # convert files to dicts containing documents that can be indexed and perform NER
    dicts, faq, qg = convert_file_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text) #zino: convert_files_to_dicts -> convert_file_to_dicts

    #write the docs to DB.
    document_store.write_documents(dicts)

    #Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question could be answered.
    retriever = TfidfRetriever(document_store=document_store)
    ranking_retriever=EmbeddingRetriever(document_store=document_store,
                                         embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                         model_format="sentence_transformers")

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Paraphrase questions by back-translation
    translator_fr = TransformersTranslator("Helsinki-NLP/opus-mt-en-fr")
    translator_en = TransformersTranslator("Helsinki-NLP/opus-mt-fr-en")

    #call generator
    qa_generator = QA_Generator(translator_fr, translator_en, retriever,ranking_retriever, reader, pipe, faq, qg)
    generated_qas = qa_generator.generate()
    num_gen = len(generated_qas)

    #write to file
    generated_qas = json.dumps(generated_qas, indent=2)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open('outputs/qg_eqBank_.json', 'w') as outfile:
        outfile.write(generated_qas)

    end = time.time()
    print("Generated a total of ", num_gen, " questions in ", int((end-start)/60), " mins"  )


class QA_Generator():
    def __init__(self, trans_fr, trans_en, retriever, ranking_retriever, reader, pipe, qa_dict, generated_questions):
        self.trans_fr = trans_fr
        self.trans_en = trans_en
        self.retriever = retriever
        self.ranking_retriever = ranking_retriever
        self.reader = reader
        self.pipe = pipe
        self.qa_dict=qa_dict

        self.original_questions=[sub['question'] for sub in qa_dict]
        self.original_answers = [sub['answer'] for sub in qa_dict]

        new_qa_dict = self.answer_finder(generated_questions)
        self.generated_questions = [sub['question'] for sub in new_qa_dict]
        self.generated_answers = [sub['answer'] for sub in new_qa_dict]

        self.translated_questions = self.back_translator(self.original_questions)
        self.summarized_answers = self.ext_summarizer(self.original_answers)

        #summary of generated answers
        self.summ_gen_answers = self.ext_summarizer(self.generated_answers)

        #translation of generated questions
        self.trans_gen_questions = self.back_translator(self.generated_questions)

    def generate(self):
        qa = list(set(self.origQue_SummAns() + self.transQue_SummAns() + self.transQue_SummAns()+\
        self.genQue_Ans() + self.genQue_SummAns() + self.transGenQue_Ans() + self.transQue_SummAns()))

        print("Ranking final pairs")
        queries = [question for question,_,_ in qa]
        answers = [answer for _,answer,_ in qa]
        embed_queries = self.ranking_retriever.embed(queries)
        embed_answers = self.ranking_retriever.embed(answers)
        scores = dot(embed_queries, embed_answers) / (norm(embed_queries) * norm(embed_answers))

        final_qa_dict = []
        for i in range(len(qa)):
            question,answer, type = qa[i]
            q_a={}
            q_a["question"]=question
            q_a["answer"]=answer
            q_a["type"] = type
            q_a["score"]=scores[i]

            #embed_query = self.ranking_retriever.embed([question])
            #embed_answer = self.ranking_retriever.embed([answer])
            #score = dot(embed_query, embed_answer) / (norm(embed_query) * norm(embed_answer))
            #q_a["score"] = score
            final_qa_dict.append(q_a)

        return final_qa_dict


    def origQue_SummAns(self):
        print("Pairing original questions with summaries of original answers")
        type= ["origQue_SummAns"] * len(self.original_questions)
        lead_1 = self.summarized_answers[0]
        lead_2 = self.summarized_answers[1]
        qa = list(map(lambda x, y, z: (x, y, z), self.original_questions, lead_1, type))
        qa_ = list(map(lambda x, y, z: (x, y, z), self.original_questions, lead_2, type))
        qa=qa+qa_

        return qa


    def transQue_OrigAns(self):
        print("Pairing back-translated questions with original answers")
        type = ["transQue_OrigAns"] * len(self.translated_questions)
        qa = list(map(lambda x, y, z: (x, y, z), self.translated_questions, self.original_answers, type))

        return qa

    def transQue_SummAns(self):
        print("Pairing translated questions with summaries of original answers")
        type = ["transQue_SummAns"] * len(self.translated_questions)
        lead_1 = self.summarized_answers[0]
        lead_2 = self.summarized_answers[1]
        qa = list(map(lambda x, y, z: (x, y, z), self.translated_questions, lead_1, type))
        qa_ = list(map(lambda x, y, z: (x, y, z), self.translated_questions, lead_2, type))
        qa=qa+qa_

        return qa

    def genQue_Ans(self):
        print("Pairing generated questions with answers")
        type = ["genQue_Ans"] * len(self.generated_questions)
        qa = list(map(lambda x, y, z: (x, y, z), self.generated_questions, self.generated_answers, type))

        return qa

    def genQue_SummAns(self):
        print("Pairing generated questions with summaries of answers")
        type = ["genQue_SummAns"] * len(self.generated_questions)
        lead_1 = self.summ_gen_answers[0]
        lead_2 = self.summ_gen_answers[1]
        qa = list(map(lambda x, y, z: (x, y, z), self.generated_questions, lead_1, type))
        qa_ = list(map(lambda x, y, z: (x, y, z), self.generated_questions, lead_2, type))
        qa=qa+qa_

        return qa

    def transGenQue_Ans(self):
        print("Pairing translated generated questions with answers")
        type = ["transGenQue_Ans"] * len(self.trans_gen_questions)
        lead_1 = self.summ_gen_answers[0]
        lead_2 = self.summ_gen_answers[1]
        qa = list(map(lambda x, y, z: (x, y, z), self.trans_gen_questions, lead_1, type))
        qa_ = list(map(lambda x, y, z: (x, y, z), self.trans_gen_questions, lead_2, type))
        qa=qa+qa_

        return qa

    def transGenQue_SummAns(self):
        print("Pairing translated generated questions with summaries of answers")
        type = ["transGenQue_SummAns"] * len(self.trans_gen_questions)
        lead_1 = self.summ_gen_answers[0]
        lead_2 = self.summ_gen_answers[1]
        qa = list(map(lambda x, y, z: (x, y,z), self.trans_gen_questions, lead_1, type))
        qa_ = list(map(lambda x, y, z: (x, y, z), self.trans_gen_questions, lead_2, type))
        qa=qa+qa_

        return qa

    def back_translator(self,text):
        print("Back-translation")
        translated = []
        for sent in text:
            paraphrase = self.trans_fr.translate(query=sent)
            paraphrase = self.trans_en.translate(query=paraphrase)
            translated.append(paraphrase.lower())
        return translated

    def answer_finder(self, questions):
        print("Generating answers for questions")
        qa_pairs = []
        for question in questions:
            current_pair={}

            # if generated question is answerable, take it
            prediction = self.pipe.run(query=question, top_k_retriever=3, top_k_reader=1)
            pred = prediction["answers"]
            if pred:  # not empty
                if pred[0]["score"] >= 1:
                    #current_pair["score"] = pred[0]["score"]
                    current_pair["question"] = question
                    answer = self.retriever.retrieve_(query=question, top_k=3)
                    answer = ' '.join(list(set(answer)))
                    answer = answer.lower()
                    current_pair["answer"]=answer
                    qa_pairs.append(current_pair)

        return qa_pairs

    def ext_summarizer(self,text):
        print("Extractive Summarization")
        summaries = []
        for sent in text:
            # extractive summarization: LEAD-1 and LEAD-2
            sents = sent_tokenize(sent, "")
            summ_1 = sents[0]
            summ_2=sents[:2]
            summaries.append(" ". join(summ_1).lower())
            summaries.append(" ".join(summ_2).lower())
        return summaries

if __name__ == "__main__":
    qa_pipeline()
