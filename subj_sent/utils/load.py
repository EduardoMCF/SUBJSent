from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import pandas as pd
import numpy as np

import itertools
from pathlib import Path
from .util import log


def load_sentences(path: str, train: bool = True) -> pd.DataFrame:
    sentences = pd.read_csv(path)

    labels = []
    if train:
        sentences, labels = sentences.text.copy(), sentences.label.copy()
    else:
        sentences = sentences.text.copy()

    return sentences, labels


@log("Load embeddings")
def load_word_embeddings(path: str, binary: bool = False, convert_to_w2v: bool = False) -> KeyedVectors:
    if convert_to_w2v:
        tmp_file = get_tmpfile("converted")

        _ = glove2word2vec(path, tmp_file)

    path = tmp_file if convert_to_w2v else path
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    return model


@log("Load embeddings")
def load_USE_embeddings(path: str):
    model = hub.load(path)
    return model
