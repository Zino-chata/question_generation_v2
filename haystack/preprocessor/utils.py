import re
import logging
import tarfile
import tempfile
import zipfile
import gzip
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Generator
import json
import sys
import neuralcoref
import pprint
from collections import Counter

from farm.data_handler.utils import http_get

from haystack.file_converter.base import BaseConverter
from haystack.file_converter.docx import DocxToTextConverter
from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.csv import CSVToTextConverter
from haystack.file_converter.tika import TikaConverter
from haystack import Document, Label
from haystack.file_converter.txt import TextConverter
from haystack.preprocessor.preprocessor import PreProcessor

logger = logging.getLogger(__name__)

###########################
from question_generation.pipelines import pipeline



def eval_data_from_json(filename: str, max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None) -> Tuple[List[Document], List[Label]]:
    """
    Read Documents + Labels from a SQuAD-style file.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents. 
    :return: (List of Documents, List of Labels)
    """

    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r") as file:
        data = json.load(file)
        if "title" not in data["data"][0]:
            logger.warning(f"No title information found for documents in QA file: {filename}")

        for document in data["data"]:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(document, preprocessor)
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)
    if len(problematic_ids) > 0:
        logger.warning(f"Could not convert an answer for {len(problematic_ids)} questions.\n"
                       f"There were conversion errors for question ids: {problematic_ids}")
    return docs, labels


def eval_data_from_jsonl(filename: str, batch_size: Optional[int] = None,
                         max_docs: Union[int, bool] = None, preprocessor: PreProcessor = None) -> Generator[Tuple[List[Document], List[Label]], None, None]:
    """
    Read Documents + Labels from a SQuAD-style file in jsonl format, i.e. one document per line.
    Document and Labels can then be indexed to the DocumentStore and be used for evaluation.

    This is a generator which will yield one tuple per iteration containing a list
    of batch_size documents and a list with the documents' labels.
    If batch_size is set to None, this method will yield all documents and labels.

    :param filename: Path to file in SQuAD format
    :param max_docs: This sets the number of documents that will be loaded. By default, this is set to None, thus reading in all available eval documents.
    :return: (List of Documents, List of Labels)
    """

    docs: List[Document] = []
    labels = []
    problematic_ids = []

    with open(filename, "r") as file:
        for document in file:
            if max_docs:
                if len(docs) > max_docs:
                    break
            # Extracting paragraphs and their labels from a SQuAD document dict
            document_dict = json.loads(document)
            cur_docs, cur_labels, cur_problematic_ids = _extract_docs_and_labels_from_dict(document_dict, preprocessor)
            docs.extend(cur_docs)
            labels.extend(cur_labels)
            problematic_ids.extend(cur_problematic_ids)

            if batch_size is not None:
                if len(docs) >= batch_size:
                    if len(problematic_ids) > 0:
                        logger.warning(f"Could not convert an answer for {len(problematic_ids)} questions.\n"
                                       f"There were conversion errors for question ids: {problematic_ids}")
                    yield docs, labels
                    docs = []
                    labels = []
                    problematic_ids = []

    yield docs, labels


def _extract_docs_and_labels_from_dict(document_dict: Dict, preprocessor: PreProcessor = None):
    docs = []
    labels = []
    problematic_ids = []

    # get all extra fields from document level (e.g. title)
    meta_doc = {k: v for k, v in document_dict.items() if k not in ("paragraphs", "title")}
    for paragraph in document_dict["paragraphs"]:
        ## Create Metadata
        cur_meta = {"name": document_dict.get("title", None)}
        # all other fields from paragraph level
        meta_paragraph = {k: v for k, v in paragraph.items() if k not in ("qas", "context")}
        cur_meta.update(meta_paragraph)
        # meta from parent document
        cur_meta.update(meta_doc)

        ## Create Document
        cur_doc = Document(text=paragraph["context"], meta=cur_meta)
        if preprocessor is not None:
            splits_dicts = preprocessor.process(cur_doc.to_dict())
            # we need to pull in _split_id into the document id for unique reference in labels
            # todo: PreProcessor should work on Documents instead of dicts
            splits = []
            offset = 0
            for d in splits_dicts:
                id = f"{d['id']}-{d['meta']['_split_id']}"
                d["meta"]["_split_offset"] = offset
                offset += len(d["text"])
                # offset correction based on splitting method
                if preprocessor.split_by == "word":
                    offset += 1
                elif preprocessor.split_by == "passage":
                    offset += 2
                else:
                    raise NotImplementedError
                mydoc = Document(text=d["text"],
                                 id=id,
                                 meta=d["meta"])
                splits.append(mydoc)
        else:
            splits = [cur_doc]
        docs.extend(splits)

        ## Assign Labels to corresponding documents
        for qa in paragraph["qas"]:
            if not qa["is_impossible"]:
                for answer in qa["answers"]:
                    ans = answer["text"]
                    ans_position = cur_doc.text[answer["answer_start"]:answer["answer_start"]+len(ans)]
                    if ans != ans_position:
                        # do not use answer
                        problematic_ids.append(qa.get("id","missing"))
                        break
                    # find corresponding document or split
                    if len(splits) == 1:
                        cur_id = splits[0].id
                        cur_ans_start = answer["answer_start"]
                    else:
                        for s in splits:
                            # If answer start offset is contained in passage we assign the label to that passage
                            if (answer["answer_start"] >= s.meta["_split_offset"]) and (answer["answer_start"] < (s.meta["_split_offset"] + len(s.text))):
                                cur_id = s.id
                                cur_ans_start = answer["answer_start"] - s.meta["_split_offset"]
                                # If a document is splitting an answer we add the whole answer text to the document
                                if s.text[cur_ans_start:cur_ans_start+len(ans)] != ans:
                                    s.text = s.text[:cur_ans_start] + ans
                                break
                    label = Label(
                        question=qa["question"],
                        answer=ans,
                        is_correct_answer=True,
                        is_correct_document=True,
                        document_id=cur_id,
                        offset_start_in_doc=cur_ans_start,
                        no_answer=qa["is_impossible"],
                        origin="gold_label",
                    )
                    labels.append(label)
            else:
                # for no_answer we need to assign each split as not fitting to the question
                for s in splits:
                    label = Label(
                        question=qa["question"],
                        answer="",
                        is_correct_answer=True,
                        is_correct_document=True,
                        document_id=s.id,
                        offset_start_in_doc=0,
                        no_answer=qa["is_impossible"],
                        origin="gold_label",
                    )
                    labels.append(label)

    return docs, labels, problematic_ids


def convert_files_to_dicts(dir_path: str, clean_func: Optional[Callable] = None, split_paragraphs: bool = False) -> \
        List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.

    :return: None
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning('Skipped file {0} as type {1} is not supported here. '
                           'See haystack.file_converter for support of more file types'.format(path, file_suffix))

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()

    documents = []
    for suffix, paths in suffix2paths.items():
        for path in paths:
            logger.info('Converting {}'.format(path))
            document = suffix2converter[suffix].convert(file_path=path, meta=None)
            text = document["text"]

            if clean_func:
                text = clean_func(text)

            if split_paragraphs:
                for para in text.split("\n\n"):
                    if not para.strip():  # skip empty paragraphs
                        continue
                    documents.append({"text": para, "meta": {"name": path.name}})
            else:
                documents.append({"text": text, "meta": {"name": path.name}})

    return documents

#zino: split text in file to sentences, convert to dict and save individual sentences each to a seperate file
def convert_file_to_dicts(dir_path: str, clean_func: Optional[Callable] = None) -> \
        List[dict]:
    """
    Convert all files(.txt, .pdf, .docx) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param dir_path: path for the documents to be written to the DocumentStore
    """

    file_paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt", ".docx", ".csv"]
    suffix2converter: Dict[str, BaseConverter] = {}

    suffix2paths: Dict[str, List[Path]] = {}
    for path in file_paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            if file_suffix not in suffix2paths:
                suffix2paths[file_suffix] = []
            suffix2paths[file_suffix].append(path)
        elif not path.is_dir():
            logger.warning('Skipped file {0} as type {1} is not supported here. '
                           'See haystack.file_converter for support of more file types'.format(path, file_suffix))

    # No need to initialize converter if file type not present
    for file_suffix in suffix2paths.keys():
        if file_suffix == ".pdf":
            suffix2converter[file_suffix] = PDFToTextConverter()
        if file_suffix == ".txt":
            suffix2converter[file_suffix] = TextConverter()
        if file_suffix == ".docx":
            suffix2converter[file_suffix] = DocxToTextConverter()
        if file_suffix == ".csv":
            suffix2converter[file_suffix] = CSVToTextConverter()

    documents = []
    new_questions = []
    original_qa_pairs=[]
    nlp_qg = pipeline("e2e-qg")
    for suffix, paths in suffix2paths.items():
        for path in paths:
            logger.info('Converting {}'.format(path))
            document = suffix2converter[suffix].convert(file_path=path, meta=None)
            text = document["text"]

            if file_suffix==".csv":
                original_qa_pairs.extend(document["meta"])

            if clean_func:
                text = clean_func(text)

            #resolve references
            text=coref_resolution(text)

            #generate questions from entire text
            new_questions.extend(nlp_qg(text))

            #generate questions from each paragraph
            paragraphs = [sub["answer"] for sub in document["meta"]]
            for paragraph in paragraphs:
                new_questions.extend(nlp_qg(paragraph))

            #sent tokenize
            sents= sent_tokenize(text,path.name)

            '''
            #identify entities
            ners, noun_chunks,_ = compute_ner_and_noun_chunks(text)
            ner=[]
            for ent, ent_type in ners:
                ent=ent.replace("\n","")
                if ent!="":
                    ner.append((ent,ent_type))
            '''

            for text in sents:
                #generate questions from each sentence
                new_questions.extend(nlp_qg(text))

                documents.append({"text": text, "meta": {"name": path.name}})

    return documents,original_qa_pairs, new_questions

"""Function that executes coreference resolution on a given text"""
def coref_resolution(text):
    nlp = SpacyMagic.load('my_english', 'en_core_web_sm', disable=[])
    #nlp = spacy.load('en')
    # Add neural coref to SpaCy's pipe
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(text)
    # fetches tokens with whitespaces from spacy document
    tok_list = list(token.text_with_ws for token in doc)
    for cluster in doc._.coref_clusters:
        # get tokens from representative cluster name
        cluster_main_words = set(cluster.main.text.split(' '))
        for coref in cluster:
            if coref != cluster.main:  # if coreference element is not the representative element of that cluster
                if coref.text != cluster.main.text and bool(set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
                    # if coreference element text and representative element text are not equal and none of the coreference element words are in representative element. This was done to handle nested coreference scenarios
                    tok_list[coref.start] = cluster.main.text + \
                        doc[coref.end-1].whitespace_
                    for i in range(coref.start+1, coref.end):
                        tok_list[i] = ""

    return "".join(tok_list)

def compute_ner_and_noun_chunks(text):
    """
    https://spacy.io/usage/linguistic-features#noun-chunks

    ents: [('today', 'DATE'), ('Patrick', 'PERSON')]
    noun_chunks: e.g. [('Autonomous cars', 'nsubj'), ('insurance liability', 'dobj')]

    :return: (ents, noun_chunks)
    """

    # spacy has memory leaks: https://github.com/explosion/spaCy/issues/3618
    nlp = SpacyMagic.load('my_english', 'en_core_web_sm', disable=[])
    doc = nlp(text)

    ents = [(ent.text, ent.label_) for ent in doc.ents]
    chunks = [(nc.text, nc.root.dep_) for nc in doc.noun_chunks]
    pos=[(d.text, d.pos_) for d in doc]

    ents = sorted(set(ents))
    chunks = sorted(set(chunks))
    pos=sorted(set(pos))

    return ents, chunks, pos

#zino: tokenize sentences (from uqa)
def sent_tokenize(raw_text, title):
    """
    :return: a list of sentences
    """
    # There are different types of sentence segmentation. See
    # https://spacy.io/usage/linguistic-features#sbd for more details
    # The sentencizer is much faster, but not as good as DependencyParser
    # Alternatively, nlp = SpacyMagic.load('en_core_web_sm')  # using DependencyParser
    #nlp = SpacyMagic.load_en_sentencizer()
    nlp = SpacyMagic.load('my_english', 'en_core_web_sm')

    text_lst = re.split(r'[\n\r]+', raw_text)
    if title and text_lst[0] == title:
        # remove the first element if is the same as the title
        text_lst = text_lst[1:]

    sentences_agg = []
    for text in text_lst:
        doc = nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        sentences_agg.extend(sentences)
    return sentences_agg

# Spacy isn't serializable but loading it is semi-expensive
class SpacyMagic(object):
    """
    Simple Spacy Magic to minimize loading time.
    >>> SpacyMagic.load("en")
    <spacy.en.English ...
    """
    _spacys = {}
    _counter = Counter()

    @classmethod
    def load(cls, name, lang, **kwargs):
        # e.g. load('my_en', 'en_core_web_sm', disable=['tagger', 'ner'])

        # spacy has memory leaks: https://github.com/explosion/spaCy/issues/3618
        # HACK: reload every K loads (via counter) to prevent leaks

        if name not in cls._spacys or cls._counter[name] % 10000 == 0:
            # only load once per thread
            import spacy
            cls._spacys[name] = spacy.load(lang, **kwargs)

        cls._counter[name] += 1
        return cls._spacys[name]

    @classmethod
    def load_en_disable_all(cls):
        name = 'en_disable_all'
        if name not in cls._spacys:
            import spacy
            nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])
            assert(len(nlp.pipeline) == 0)
            assert(len(nlp.pipe_names) == 0)
            cls._spacys[name] = nlp
        return cls._spacys[name]

    @classmethod
    def load_en_sentencizer(cls):
        # https://spacy.io/usage/linguistic-features#sbd-component
        name = 'en_sentencizer'
        if name not in cls._spacys:
            # only load once per thread
            from spacy.lang.en import English
            nlp = English()
            sbd = nlp.create_pipe('sentencizer')   # or: nlp.create_pipe('sbd')
            nlp.add_pipe(sbd)
            cls._spacys[name] = nlp
        return cls._spacys[name]

def tika_convert_files_to_dicts(
        dir_path: str,
        clean_func: Optional[Callable] = None,
        split_paragraphs: bool = False,
        merge_short: bool = True,
        merge_lowercase: bool = True
) -> List[dict]:
    """
    Convert all files(.txt, .pdf) in the sub-directories of the given path to Python dicts that can be written to a
    Document Store.

    :param merge_lowercase: allow conversion of merged paragraph to lowercase
    :param merge_short: allow merging of short paragraphs
    :param dir_path: path for the documents to be written to the DocumentStore
    :param clean_func: a custom cleaning function that gets applied to each doc (input: str, output:str)
    :param split_paragraphs: split text in paragraphs.

    :return: None
    """
    converter = TikaConverter()
    paths = [p for p in Path(dir_path).glob("**/*")]
    allowed_suffixes = [".pdf", ".txt"]
    file_paths: List[Path] = []

    for path in paths:
        file_suffix = path.suffix.lower()
        if file_suffix in allowed_suffixes:
            file_paths.append(path)
        elif not path.is_dir():
            logger.warning('Skipped file {0} as type {1} is not supported here. '
                           'See haystack.file_converter for support of more file types'.format(path, file_suffix))

    documents = []
    for path in file_paths:
        logger.info('Converting {}'.format(path))
        document = converter.convert(path)
        meta = document["meta"] or {}
        meta["name"] = path.name
        text = document["text"]
        pages = text.split("\f")

        if split_paragraphs:
            if pages:
                paras = pages[0].split("\n\n")
                # pop the last paragraph from the first page
                last_para = paras.pop(-1) if paras else ''
                for page in pages[1:]:
                    page_paras = page.split("\n\n")
                    # merge the last paragraph in previous page to the first paragraph in this page
                    if page_paras:
                        page_paras[0] = last_para + ' ' + page_paras[0]
                        last_para = page_paras.pop(-1)
                        paras += page_paras
                if last_para:
                    paras.append(last_para)
                if paras:
                    last_para = ''
                    for para in paras:
                        para = para.strip()
                        if not para:
                            continue
                        # merge paragraphs to improve qa
                        # merge this paragraph if less than 10 characters or 2 words
                        # or this paragraph starts with a lower case and last paragraph does not end with a punctuation
                        if merge_short and len(para) < 10 or len(re.findall(r'\s+', para)) < 2 \
                                or merge_lowercase and para and para[0].islower() and last_para \
                                and last_para[-1] not in r'.?!"\'\]\)':
                            last_para += ' ' + para
                        else:
                            if last_para:
                                documents.append({"text": last_para, "meta": meta})
                            last_para = para
                    # don't forget the last one
                    if last_para:
                        documents.append({"text": last_para, "meta": meta})
        else:
            if clean_func:
                text = clean_func(text)
            documents.append({"text": text, "meta": meta})

    return documents


def fetch_archive_from_http(url: str, output_dir: str, proxies: Optional[dict] = None):
    """
    Fetch an archive (zip or tar.gz) from a url via http and extract content to an output directory.

    :param url: http address
    :type url: str
    :param output_dir: local path
    :type output_dir: str
    :param proxies: proxies details as required by requests library
    :type proxies: dict
    :return: bool if anything got fetched
    """
    # verify & prepare local directory
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)

    is_not_empty = len(list(Path(path).rglob("*"))) > 0
    if is_not_empty:
        logger.info(
            f"Found data stored in `{output_dir}`. Delete this first if you really want to fetch new data."
        )
        return False
    else:
        logger.info(f"Fetching from {url} to `{output_dir}`")

        # download & extract
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)  # making tempfile accessible
            # extract
            if url[-4:] == ".zip":
                zip_archive = zipfile.ZipFile(temp_file.name)
                zip_archive.extractall(output_dir)
            elif url[-7:] == ".tar.gz":
                tar_archive = tarfile.open(temp_file.name)
                tar_archive.extractall(output_dir)
            elif url[-3:] == ".gz":
                filename = url.split("/")[-1].replace(".gz", "")
                output_filename = Path(output_dir) / filename
                with gzip.open(temp_file.name) as f, open(output_filename, "wb") as output:
                        for line in f:
                               output.write(line)
            else:
                logger.warning('Skipped url {0} as file type is not supported here. '
                               'See haystack documentation for support of more file types'.format(url))
            # temp_file gets deleted here
        return True


def squad_json_to_jsonl(squad_file: str, output_file: str):
    """
    Converts a SQuAD-json-file into jsonl format with one document per line.

    :param squad_file: SQuAD-file in json format.
    :type squad_file: str
    :param output_file: Name of output file (SQuAD in jsonl format)
    :type output_file: str
    """
    with open(squad_file) as json_file, open(output_file, "w") as jsonl_file:
        squad_json = json.load(json_file)

        for doc in squad_json["data"]:
            json.dump(doc, jsonl_file)
            jsonl_file.write("\n")
