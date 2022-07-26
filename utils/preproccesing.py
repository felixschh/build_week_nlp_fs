import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
import re
import string

nlp = spacy.load("en_core_web_sm")
fasttext = FastText("simple")

def preprocessing(sentence):
    doc = nlp(sentence)

    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop
              and not token.is_currency and not token.like_email and not token.like_url and not token.is_digit
              and token.is_ascii]
    return tokens

def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def  clean_text(text):
    # text =  text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    # text = re.sub(r"[-()"#\@;:<>{}`+=~|.!?,]"", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\W)"," ",text) 
    text = re.sub('\S\d\S\s*','', text)

    return text

def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]

def fit_comment():
    return