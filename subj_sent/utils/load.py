from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import pandas as pd
import numpy as np

import itertools

def load_sentences(path : str, train : bool = True) - > pd.DataFrame:
    sentences = pd.read_csv(path).sample(frac = 1) # loads and shuffle the dataframe
    
    labels = []
    if train:
        sentences, labels = sentences.text.copy(), sentences.label.copy()

    return sentences, labels

def load_word_embeddings(path : str, binary : bool = False, convert_to_w2v : bool = False) -> KeyedVectors:
    if convert_to_w2v:
        glove_file = datapath(f'{path}.{txt if not binary else bin}')
        tmp_file = get_tmpfile(f"converted.{txt if not binary else bin}")

        _ = glove2word2vec(glove_file, tmp_file)

    path = tmp_file if convert_to_w2v else path
    model = KeyedVectors.load_word2vec_format(path ,binary = binary)
    return model

def load_USE_embeddings(path : str):
    model = hub.load(path)
    return model
    
    

    


