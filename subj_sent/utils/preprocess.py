import gc
import itertools
import re
import string
from time import time
from typing import List

import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk import tokenize

from .util import log

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(sentence: str,
               mode: str = 'word',
               remove_stopwords: bool = False) -> str:

    if mode == 'word':
        sentence = sentence.lower()
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
        sentence = re.sub(r"\'s", " \'s", sentence)
        sentence = re.sub(r"\'ve", " \'ve", sentence)
        sentence = re.sub(r"n\'t", " n\'t", sentence)
        sentence = re.sub(r"\'re", " \'re", sentence)
        sentence = re.sub(r"\'d", " \'d", sentence)
        sentence = re.sub(r"\'ll", " \'ll", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"\(", " \( ", sentence)
        sentence = re.sub(r"\)", " \) ", sentence)
        sentence = re.sub(r"\?", " \? ", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence)

    stopwords = set()
    if remove_stopwords:
        stopwords = set(nltk.corpus.stopwords.words('english'))

    tokens = [token for token in sentence.split() if token not in stopwords]
    return tokens if mode == 'word' else ' '.join(tokens)

def parse_text(text : str, mode : str, max_length : int = 200, remove_stopwords: bool = False):
    text = tokenize.sent_tokenize(text)
    sentences = []
    for sentence in text:
        preprocessed_sentence = preprocess(sentence, mode, remove_stopwords)
        if mode == 'word':
            sentences.extend(_pad_sentence(preprocessed_sentence, max_length))
        else:
            sentences.append(preprocessed_sentence)

    return sentences

def get_word_embeddings_from_text(text : List[List[str]], model: KeyedVectors, embeddings_dim: int) -> List[str]:
    embeddings = []
    np.random.seed(777)
    base_embedding = np.random.uniform(-0.25, 0.25, embeddings_dim)
    for sentence in text:
        for word in sentence:
            if type(word) is list:
                print(word)
                print(text)
            if word in model.vocab:
                embeddings.append(np.array(model.word_vec(word)))
            else:
                embeddings.append(base_embedding)
    
    return np.array(embeddings)


@log("Generate word embeddings")
def get_word_embeddings(sentences: list, model: KeyedVectors, embeddings_dim: int, max_length: int) -> List[str]:
    indexed_sentences, _, vocabulary_inverse = _parse_data_word_embeddings(sentences, max_length)
    embeddings = _generate_embeddings(
        vocabulary_inverse, model, embeddings_dim)
    del model
    gc.collect()

    sentence_embeddings = np.array([[embeddings[word] for word in sentence]
                                    for sentence in indexed_sentences])
    del embeddings
    gc.collect()
    return sentence_embeddings

def get_sentence_embeddings_from_text(text : List[str], model : object):
    if type(text) is str:
        print('teste')
        text = [text]
    return np.array(model(text))

@log("Generate sentence embeddings")
def get_sentence_embeddings(sentences: list, model: object):
    partitions = list(range(0, len(sentences), 1000)) + [len(sentences)]
    chunks = [partitions[i:i+2] for i in range(len(partitions)-1)]

    res = []
    for chunk in chunks:
        res.append(np.array(model(sentences[chunk[0]:chunk[1]])))

    sentence_embeddings = np.concatenate(res)

    return sentence_embeddings

def _pad_sentence(sentence : str, max_length : int, padding_symbol : str = '<PAD/>') -> List[List]:
    if len(sentence) > max_length:
        sentence = [sentence[idx:idx+max_length] for idx in range(0, len(sentence), max_length)]
        if len(sentence[-1]) < max_length:
            sentence[-1] = sentence[-1] + [padding_symbol] * (max_length - len(sentence[-1]))
    else:
        sentence = [sentence + [padding_symbol] * (max_length - len(sentence))]

    return sentence    

def _pad_sentences(sentences: list, max_length: int, padding_symbol='<PAD/>'):
    padded_sentences = np.array([sentence + [padding_symbol] * (max_length - len(sentence))
                                 if len(sentence) <= max_length else sentence[:max_length]
                                                                for sentence in sentences])
    return padded_sentences


def _build_vocabulary(sentences: list) -> tuple:
    words = set(itertools.chain(*sentences))
    vocabulary_inverse = list(words)
    vocabulary = {word: index for index, word in enumerate(vocabulary_inverse)}
    return vocabulary, vocabulary_inverse


def _parse_data_word_embeddings(sentences: list, pad_length: int) -> tuple:
    padded_sentences = _pad_sentences(sentences, max_length=pad_length)
    vocabulary, vocabulary_inverse = _build_vocabulary(padded_sentences)
    indexed_sentences = np.array([[vocabulary[word] for word in sentence]
                                  for sentence in padded_sentences])
    return indexed_sentences, vocabulary, vocabulary_inverse


def _generate_embeddings(vocabulary_inverse: list, model: KeyedVectors, embeddings_dim: int) -> dict:
    np.random.seed(777)
    base_embedding = np.random.uniform(-0.25, 0.25, embeddings_dim)

    embeddings = {}
    for id, word in enumerate(vocabulary_inverse):
        if word in model.vocab:
            embeddings[id] = np.array(model.word_vec(word))
        else:
            embeddings[id] = base_embedding

    return embeddings
