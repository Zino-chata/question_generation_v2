import logging
import csv
import sys
import re
from pathlib import Path
from typing import List, Optional, Any, Dict

from haystack.file_converter.base import BaseConverter

logger = logging.getLogger(__name__)


class CSVToTextConverter(BaseConverter):
    def __init__(self, remove_numeric_tables: Optional[bool] = False, valid_languages: Optional[List[str]] = None):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """

        super().__init__(remove_numeric_tables=remove_numeric_tables,
                         valid_languages=valid_languages)

    def convert(self,
                file_path: Path,
                meta: Optional[Dict[str, str]] = None,
                encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Reads text from a csv file and executes optional preprocessing steps.

        :param file_path: Path of the file to convert
        :param meta: Optional meta data that should be associated with the the document (e.g. name)
        :param encoding: Encoding of the file

        :return: Dict of format {"text": "The text from file", "meta": meta}}

        """
        count=0
        text=[]
        orig_data=[]
        with open(file_path) as f:
            csvreader = csv.reader(f, delimiter=',')
            for row in csvreader:
                if count>0:
                    data = {}
                    data["question"] = row[1].lower()
                    data["question"] = re.sub("\?|\!|,|\.|\\\|\\/|\\(|\\)|\-", " ", data["question"])
                    data["question"] = re.sub("\’|\'|\\\"", "", data["question"])
                    data["question"] = " ".join(data["question"].split())

                    cleaned_ans=remove_html_tags(row[2]).replace(":"," ").replace(";"," ")
                    data["answer"]=clean_sent(row[2])
                    text.append(cleaned_ans)
                    orig_data.append(data)
                count=count+1


        text=' '.join(text)
        pages=text.split("\f")

        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if self.remove_numeric_tables:
                    if words and len(digits) / len(words) > 0.4 and not line.strip().endswith("."):
                        logger.debug(f"Removing line '{line}' from {file_path}")
                        continue

                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if self.valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for {file_path} is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "".join(pages)
        document = {"text": text, "meta": orig_data}
        return document

#clean
def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def clean_sent(v):
    v = remove_html_tags(v)
    v = re.sub("\+", " plus ", v)
    v = re.sub("&", " and ", v)
    v = re.sub("%", " percentage ", v)
    v = re.sub(" $", " dollars ", v)
    v = re.sub("$ ", " dollars ", v)
    v = re.sub("#", " number ", v)
    v = re.sub("\n", " , ", v)
    v = re.sub("\||,|\!|\\\|\\/|\\(|\\)|\\-|\\—|\:|\;|\\“|\\”", " ", v)
    v = re.sub("\’|\'|\\\"", " ", v)
    #v = re.sub("[0-9]+[.|,]?-![0-9]+", " value_label ", v)
    v = " ".join(v.split())
    v = re.sub(", ,", " , ", v)
    v = " ".join(v.lower().split())
    return v