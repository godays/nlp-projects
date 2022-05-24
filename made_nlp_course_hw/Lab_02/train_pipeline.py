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
import tqdm


from nltk.tokenize import WordPunctTokenizer
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


def train(
    model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None
):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        history.append(loss.cpu().data.numpy())
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train_model(
    train_data,
    valid_data,
    test_data,
    SRC,
    TRG,
    encoder,
    decoder,
    seq2seq,
    model_name,
    batch_size=32,
    n_iter=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _len_sort_key(x):
        return len(x.src)

    BATCH_SIZE = batch_size

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
        sort_key=_len_sort_key,
    )

    Encoder = encoder
    Decoder = decoder
    Seq2Seq = seq2seq

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param, -0.08, 0.08)

    model.apply(init_weights)

    PAD_IDX = TRG.vocab.stoi["<pad>"]
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_history = []
    valid_history = []
    N_EPOCHS = n_iter
    CLIP = 1
    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        train_loss = train(
            model,
            train_iterator,
            optimizer,
            criterion,
            CLIP,
            train_history,
            valid_history,
        )
        valid_loss = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{model_name}.pt")

        train_history.append(train_loss)
        valid_history.append(valid_loss)

    return model, train_history, valid_history, test_iterator
