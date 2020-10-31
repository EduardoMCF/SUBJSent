import itertools
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from sklearn.model_selection import train_test_split

from .util import log


def load_sentences(path: str, train: bool = True) -> pd.Series:
    sentences = pd.read_csv(path)

    labels = []
    if train:
        sentences, labels = sentences.text.copy(), sentences.label.copy()
    else:
        sentences = sentences.Comment.copy()

    return sentences, labels

@log('Load KeyedVector')
def load_word_embeddings(path : str) -> KeyedVectors:
    model = KeyedVectors.load(path, mmap='r')
    return model

@log('Create KeyedVector')
def save_word_embeddings(filename : str, path_to_model: str, binary: bool = False, convert_to_w2v: bool = False):
    model = load_raw_word_embeddings(path_to_model, binary, convert_to_w2v)
    
    embeddings_path = Path(__file__).parent.parent.parent / 'embeddings'
    if not os.path.exists(embeddings_path):
        os.mkdir(embeddings_path)

    model.save(embeddings_path / f'{filename}.kv')

@log("Load embeddings")
def load_raw_word_embeddings(path: str, binary: bool = False, convert_to_w2v: bool = False) -> KeyedVectors:
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
