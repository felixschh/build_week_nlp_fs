import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import FastText
import re
import string

np.random.seed(0)

nlp = spacy.load("en_core_web_sm")
fasttext = FastText("simple")


def load_data(pth):
    df = pd.read_csv(pth)
    df.drop(columns=["id"], inplace=True)
    return df


def train_test_split(df, train_size=0.8):

    df_idx = [i for i in range(len(df))]
    np.random.shuffle(df_idx)

    len_train = int(len(df) * train_size)
    df_train = df.iloc[:len_train].reset_index(drop=True)
    df_test = df.iloc[len_train:].reset_index(drop=True)
    return df_train, df_test


def preprocessing(sentence):
    """
    Get rid of punctuations, space, stop words, currency symbols,
    email-alike sequences, url-alike sequences numbers and non-ascii characters.
    params sentence: a str containing the sentence we want to preprocess
    return the tokens list
    """
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


def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]


df = load_data('jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')
# print(df.head())
train_df, test_df = train_test_split(df)
# print(len(train_df), len(test_df))

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

# data = clean_text(df['comment_text'])
# print(data)

df['comment_text'] = df['comment_text'].apply(lambda x:clean_text(x))
print(df)



'''Resume from here'''

class TrainData(Dataset):
    def __init__(self, df, max_seq_len=32):
        self.max_seq_len = max_seq_len

        train_iter = iter(df.comment_text.values)
        self.vec = FastText("simple")
        # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0])
        # replacing the vector associated with 0 (unknown) to become zeros
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0])
        self.vectorizer = lambda x: self.vec.vectors[x]
        
        sequences = [padding(encoder(preprocessing(
            sequence), self.vec), max_seq_len) for sequence in df.comment_text.tolist()]
        self.sequences = sequences

        self.labels = df.drop(columns='comment_text').values        # target
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]


train_df, test_df = train_test_split(df)
train_df, test_df = TrainData(train_df), TrainData(test_df)

def collate_train(batch, vectorizer=train_df.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor(np.array([item[1] for item in batch]))
    return inputs, target

def collate_test(batch, vectorizer=test_df.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor(np.array([item[1] for item in batch]))
    return inputs, target


train_loader = DataLoader(train_df, batch_size=32, collate_fn=collate_train, shuffle=True)
test_loader = DataLoader(test_df, batch_size=32, collate_fn=collate_test)