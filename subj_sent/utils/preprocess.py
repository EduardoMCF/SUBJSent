import numpy as np
import nltk
from gensim.models import KeyedVectors

import re
import gc
import string
import itertools
from time import time
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


@log("Generate word embeddings")
def get_word_embeddings(sentences: list, model: KeyedVectors, embeddings_dim: int, pad_length: int = None):
    indexed_sentences, _, vocabulary_inverse = _parse_data_word_embeddings(
        sentences, pad_length)
    embeddings = _generate_embeddings(
        vocabulary_inverse, model, embeddings_dim)
    del model
    gc.collect()

    sentence_embeddings = np.array([[embeddings[word] for word in sentence]
                                    for sentence in indexed_sentences])
    del embeddings
    gc.collect()
    return sentence_embeddings


@log("Generate sentence embeddings")
def get_sentence_embeddings(sentences: list, model: object):
    partitions = list(range(0, len(sentences), 1000)) + [len(sentences)]
    chunks = [partitions[i:i+2] for i in range(len(partitions)-1)]

    res = []
    for chunk in chunks:
        res.append(np.array(model(sentences[chunk[0]:chunk[1]])))
    sentence_embeddings = np.concatenate(res)
    return sentence_embeddings


def _pad_sentences(sentences: list, padding_symbol='<PAD/>', max_length: int = None):
    max_length = max(len(sentence)
                     for sentence in sentences) if not max_length else max_length
    padded_sentences = np.array([sentence + [padding_symbol] * (max_length - len(sentence))
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
    embeddings_concatenated = np.concatenate(
        [model.word_vec(word) for word in list(model.vocab.keys())[:10**6]])
    embeddings_variance = np.var(embeddings_concatenated, ddof=1)
    base_embedding = np.random.uniform(-embeddings_variance,
                                       embeddings_variance, embeddings_dim)

    embeddings = {}
    for id, word in enumerate(vocabulary_inverse):
        if word in model.vocab:
            embeddings[id] = np.array(model.word_vec(word))
        else:
            embeddings[id] = base_embedding

    return embeddings
