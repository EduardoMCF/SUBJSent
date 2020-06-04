from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import pandas as pd
import numpy as np

import itertools
from pathlib import Path

def load_sentences(path : str, train : bool = True) -> pd.DataFrame:
    print('load/load_sentences')
    sentences = pd.read_csv(path)
    
    labels = []
    if train:
        sentences, labels = sentences.text.copy(), sentences.label.copy()
    else:
        sentences = sentences.text.copy()

    return sentences, labels

def load_word_embeddings(path : str, binary : bool = False, convert_to_w2v : bool = False) -> KeyedVectors:
    print('Loading embeddings')
    if convert_to_w2v:
        print(str(Path(path).parent / "converted"))
        tmp_file = get_tmpfile("converted")

        _ = glove2word2vec(path, tmp_file)
        print(tmp_file)

    path = tmp_file if convert_to_w2v else path
    model = KeyedVectors.load_word2vec_format(path ,binary = binary)
    return model

def load_USE_embeddings(path : str):
    print('Loading embeddings')
    model = hub.load(path)
    return model
    
    

    


