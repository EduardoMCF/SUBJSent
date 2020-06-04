from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras import optimizers

from subj_sent import models
from subj_sent import utils

import os
from pathlib import Path
from functools import partial


models_dict = {"MLP" : models.MLP, "CNN" :  models.CNN}
optimizers = {'adam' : optimizers.Adam, 'sgd' : optimizers.SGD,
              'rmsprop' : optimizers.RMSprop, 'adadelta' : optimizers.Adadelta,
              'adagrad' : optimizers.Adagrad, 'adamax' : optimizers.Adamax,
              'nadam' : optimizers.Nadam}


def train(data_path : str, load_model : bool, model_path : str, model_params : dict,
          embeddings_params : dict, train_params : dict, preprocess_params : dict) -> list:

    print('facade/train')
    
    sentences_with_embeddings, labels = _data_preprocess(data_path, True, embeddings_params,
                                                                          preprocess_params)
    input_shape = sentences_with_embeddings[0].shape
    max_emb,min_emb = max(sentences_with_embeddings, key = lambda e : e.shape).shape, min(sentences_with_embeddings, key = lambda e : e.shape).shape
    print(f"Model input shape: {sentences_with_embeddings.shape} {input_shape}, {max_emb}, {min_emb}")
    
    model_params['params']['optimizer'] = optimizers[model_params['params']['optimizer'].lower()]
    model = _create_model(input_shape, load_model, model_path, model_params)

    X_train, X_test, y_train, y_test = train_test_split(sentences_with_embeddings, labels,
                                                        test_size = train_params['test_size'],
                                                        random_state = 777)
    print(f'X_train : {X_train.shape}, \ny_train : {y_train.shape}, \nX_test : {X_test.shape}, y_test : {y_test.shape}')

    from collections import Counter
    print('Counter',Counter(y_train), Counter(y_test))
    
    history = model.train(X_train, y_train,
                          epochs = train_params['epochs'],
                          batch_size = train_params['batch_size'],
                          validation_split = train_params['validation_size'])
    
    evaluate = model.evaluate(X_test, y_test)

    save_path = 'saved_models/'
    save_path, count = _save_file_name(save_path)
    model.save(f"{save_path}/model {count}.h5")

    metrics = model_params['params']['metrics']
    with open(f"{save_path}/results model {count}.txt", 'w') as results:
        results.write('\n'.join(f"{name} = {metric}" for name, metric in zip(['loss'] + metrics, evaluate)))

    if train_params['plot_history']:
        _plot_history(history, f"{save_path}/plots model {count}")
    
    return ['loss']+metrics, evaluate

def predict(data_path : str, model_path : str, model_params : dict, embeddings_params : dict,
                                                                    preprocess_params : dict):

    model = _create_model(None, True, model_path, model_params)
    input_shape = model._input_shape
    
    sentences_with_embeddings, _ = _data_preprocess(data_path, False, embeddings_params,
                                                      preprocess_params, input_shape[0])
    

    print(f"Model input shape: {input_shape}, {sentences_with_embeddings.shape}")
    
   
    predictions = pd.Series(model.predict(sentences_with_embeddings).ravel())

    save_path = 'data_output/'
    save_path, count = _save_file_name(save_path)
    predictions.to_csv(f"{save_path}/data {count}.csv", index = False)

def _data_preprocess(data_path : str, train : bool, embeddings_params : dict, preprocess_params : dict,
                                                                  padding_length : int = None) -> list:

    print('facade/data_preprocess')
    sentences, labels = utils.load_sentences(data_path, train)
    preprocess_params['mode'] = embeddings_params['type']
    preprocess = partial(utils.preprocess,**preprocess_params)
    sentences = sentences.apply(preprocess)
    
    print(sentences[0])

    if embeddings_params["type"] == "word":
        model = utils.load_word_embeddings(embeddings_params['path'],
                                           embeddings_params['binary'],
                                           embeddings_params['convert_to_w2v'])
        sentences_with_embeddings = utils.get_word_embeddings(sentences, model,
                                                              embeddings_params['length'],
                                                              padding_length)
    elif embeddings_params["type"] == "sentence":
        model = utils.load_USE_embeddings(embeddings_params['path'])
        print('finished/load')
        sentences_with_embeddings = utils.get_sentence_embeddings(sentences, model)
        print('finished/get_sent')
        print('shape',sentences_with_embeddings.shape)
    else:
        raise ValueError("Embedding type must be 'word' or 'sentence'.")
    
    return sentences_with_embeddings, labels

def _create_model(input_shape : tuple, load_model : bool, model_path : str, model_params : dict):
    print('facade/create_model')
    model_type = model_params['type']
    #model_params = {} if load_model else model_params
    if model_type == 'MLP':
        del model_params['params']['num_filters'] ; del model_params['params']['kernel_sizes']

    model = models_dict[model_type]
    model = model.load(model_path) if load_model else model(input_shape,**model_params['params'])
    
    return model

def _plot_history(history : object, path : str):
    print('facade/plot_history')
    metrics_dict = history.history
    metrics_names = list(filter(lambda s: not s.startswith('val'), metrics_dict.keys()))

    fig, ax = plt.subplots(nrows=len(metrics_names), ncols=1)
    for i,metric in enumerate(metrics_names):
        ax[i].plot(metrics_dict[metric])
        ax[i].plot(metrics_dict[f"val_{metric}"])
        ax[i].set_title(f'model {metric}')
        ax[i].set_ylabel(metric)
        ax[i].set_xlabel('epoch')
        ax[i].legend(['train', 'validation'], loc='upper left')

    fig.tight_layout()    
    fig.savefig(f"{path}.png")
    
def _save_file_name(path : str) -> str:
    print('facade/save_file_name')
    path = Path(__file__).parent.parent / path
    print('p',path)
    files = os.listdir(path)
    count = max(map(lambda name: int(name.split(' ')[-1].split('.')[0]),files)) + 1 if files else 1
    return path,count
