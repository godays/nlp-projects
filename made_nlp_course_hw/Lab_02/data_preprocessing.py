import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

import spacy

import random
import math
import time


from nltk.tokenize import WordPunctTokenizer
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


tokenizer_W = WordPunctTokenizer()
PATH_TO_DATA = "data.txt"


def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def return_preprocessed_data():
    SRC = Field(tokenize=tokenize, init_token="<sos>", eos_token="<eos>", lower=True)

    TRG = Field(tokenize=tokenize, init_token="<sos>", eos_token="<eos>", lower=True)

    dataset = TabularDataset(
        path=PATH_TO_DATA, format="tsv", fields=[("trg", TRG), ("src", SRC)]
    )

    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    SRC.build_vocab(train_data, min_freq=3)
    TRG.build_vocab(train_data, min_freq=3)
    return train_data, valid_data, test_data, SRC, TRG
