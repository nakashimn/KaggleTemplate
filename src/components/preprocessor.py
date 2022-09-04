import os
import codecs
import pandas as pd
import nltk
from nltk.corpus import stopwords
from text_unidecode import unidecode
import traceback

def replace_encoding_with_utf8(error: UnicodeError):
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError):
    return error.object[error.start : error.end].decode("cp1252"), error.end

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def train_dataset(self):
        df_train = pd.read_csv(self.config["path"]["traindata"])
        return df_train

    def test_dataset(self):
        df_test = pd.read_csv(self.config["path"]["testdata"])
        return df_test

    def pseudo_dtaset(self):
        # Inplement
        return None

class TextCleaner:
    def __init__(self):
        pass

    def resolve_encodings_and_normalize(self, text: str) -> str:
        text = (
            text.encode("raw_unicode_escape")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
            .encode("cp1252", errors="replace_encoding_with_utf8")
            .decode("utf-8", errors="replace_decoding_with_cp1252")
        )
        text = unidecode(text)
        return text

    def clean(self, data, col):
        # Resolve encode
        data[col] = data[col].apply(self.resolve_encodings_and_normalize)
        # Replace Upper to Lower
        data[col] = data[col].str.lower()
        # Replace unicode
        data[col] = data[col].str.replace(r"\n{3,}", "\n\n", regex=True)
        data[col] = data[col].str.replace(r"\t+", " ", regex=True)
        data[col] = data[col].str.replace("\xa0", " ", regex=True)
        # Replace
        data[col] = data[col].str.replace(r"what's", "what is ", regex=True)
        data[col] = data[col].str.replace(r"\'ve", " have ", regex=True)
        data[col] = data[col].str.replace(r"can't", "cannot ", regex=True)
        data[col] = data[col].str.replace(r"n't", " not ", regex=True)
        data[col] = data[col].str.replace(r"i'm", "i am ", regex=True)
        data[col] = data[col].str.replace(r"\'re", " are ", regex=True)
        data[col] = data[col].str.replace(r"\'d", " would ", regex=True)
        data[col] = data[col].str.replace(r"\'ll", " will ", regex=True)
        data[col] = data[col].str.replace(r"\'scuse", " excuse ", regex=True)
        data[col] = data[col].str.replace(r"\'s", " ", regex=True)
        # Remove
        data[col] = data[col].str.replace('(', ' ', regex=True)
        data[col] = data[col].str.replace(')', ' ', regex=True)
        data[col] = data[col].str.replace(':', ' ', regex=True)
        data[col] = data[col].str.replace(';', ' ', regex=True)
        data[col] = data[col].str.replace('[', ' ', regex=True)
        data[col] = data[col].str.replace(']', ' ', regex=True)
        # Clean some punctutations
        data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3', regex=True)
        # Replace repeating characters more than 3 times to length of 3
        data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1', regex=True)
        # Add space around repeating characters
        data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ', regex=True)
        # patterns with repeating characters
        data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1', regex=True)
        data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1', regex=True)
        data[col] = data[col].str.replace(r'[ ]{2,}',' ', regex=True).str.strip()
        data[col] = data[col].str.replace(r'[ ]{2,}',' ', regex=True).str.strip()
        # for tokenize
        data[col] = data[col].str.replace(r'\"', ' \" ', regex=True)
        data[col] = data[col].str.replace(r', ', ' , ', regex=True)
        data[col] = data[col].str.replace(r'\.', ' . ', regex=True)
        data[col] = data[col].str.replace(r'\n', ' [SEP] ', regex=True)
        # avoid duplicated spaces
        data[col] = data[col].str.replace(r' +', ' ', regex=True)
        data[col] = data[col].str.strip()
        return data
