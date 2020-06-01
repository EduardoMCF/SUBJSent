import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

import string

def preprocess(sentence : str,
               remove_punctuations : bool = False,
               remove_digits : bool = False,
               remove_stopwords : bool = False) -> str:

    translate_elements = []
    if remove_punctuations:
        translate_elements += string.punctuation
    if remove_digits:
        translate_elements += string.digits
    sentence = sentence.translate(str.maketrans("", "", translate_elements))

    stopwords = set()
    if remove_stopwords:
        stopwords = set(stopwords.words('english'))

    tokens = [token for token in nltk.word_tokenize(sentence) if token not in stopwords]
    return ' '.join(tokens)

def _pad_sentences(sentences : list, padding_symbol = '<PAD/>'):
    max_length = max(len(sentence) for sentence in sentences)
    padded_sentences = np.array([sentence + [padding_symbol] * (max_length - len(sentence)) 
                                                        for sentence in sentences])
    return padded_sentences

def _build_vocabulary(sentences : list) -> tuple:
    words = set(itertools.chain(*sentences))
    vocabulary_inverse = list(words)
    vocabulary = {word : index for index, word in enumerate(vocabulary_inverse)}
    return vocabulary, vocabulary_inverse

def _parse_data_word_embeddings(sentences : list) -> tuple:
    padded_sentences = _pad_sentences(sentences)
    vocabulary, vocabulary_inverse = _build_vocabulary(padded_sentences)
    indexed_sentences = np.array([[vocabulary[word] for word in sentence] 
                                        for sentence in padded_sentences])
    return indexed_sentences, vocabulary, vocabulary_inverse

def _generate_embeddings(vocabulary_inverse : list, model : KeyedVectors, embeddings_dim : int) -> dict:
    embeddings = {id : (model.word_vec(word) if word in model.vocab else -1) 
                               for id, word in enumerate(vocabulary_inverse)}

    valid_embeddings = np.concatenate([embedding for embedding in embeddings.values()])
    embeddings_variance = np.var(valid_embeddings.ravel(), ddof=1)
    base_embedding = np.random.uniform(-embeddings_variance, embeddings_variance, embeddings_dim)
    for id, embedding in embeddings.items():
        if embedding == -1:
            embeddings[id] = base_embedding

    return embeddings

def get_word_embeddings(sentences : list, model : KeyedVectors, embeddings_dim : int):
    indexed_sentences, _, vocabulary_inverse = _parse_data_word_embeddings(sentences)
    embeddings = _generate_embeddings(vocabulary_inverse, model, embeddings_dim)

    sentence_embeddings = np.stack([np.stack([embeddings[word] for word in sentence])
                                                  for sentence in indexed_sentences])
    
    return sentence_embeddings

def get_sentence_embeddings(sentences : list, model : object):
    sentence_embeddings = model(sentences)
    return sentence_embeddings
    











     
    


    

