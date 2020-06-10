from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import optimizers

from subj_sent import models
from subj_sent import utils

from functools import partial


models_dict = {"MLP" : models.MLP, "CNN" :  models.CNN}
optimizers = {'adam' : optimizers.Adam, 'sgd' : optimizers.SGD,
              'rmsprop' : optimizers.RMSprop, 'adadelta' : optimizers.Adadelta,
              'adagrad' : optimizers.Adagrad, 'adamax' : optimizers.Adamax,
              'nadam' : optimizers.Nadam}


def train(data_path : str, model_path : str, model_params : dict,
          embeddings_params : dict, train_params : dict, preprocess_params : dict) -> list:
 
    sentences_with_embeddings, labels = _data_preprocess(data_path, True, embeddings_params,
                                                                          preprocess_params)
    input_shape = sentences_with_embeddings[0].shape
    model_params['params']['optimizer'] = optimizers[model_params['params']['optimizer'].lower()]
    model = _create_model(input_shape, False, model_path, model_params)

    X_train, X_test, y_train, y_test = train_test_split(sentences_with_embeddings, labels,
                                                        test_size = train_params['test_size'],
                                                        random_state = 777)
    
    history = model.train(X_train, y_train,
                          epochs = train_params['epochs'],
                          batch_size = train_params['batch_size'],
                          validation_split = train_params['validation_size'])
    
    evaluate = model.evaluate(X_test, y_test)

    save_path = 'saved_models/'
    save_path, count = utils.savefile_name(save_path)
    model_path = save_path / f"model.h5"
    results_path = save_path / f"results.txt"
    plot_path = save_path / f"plots"

    model.save(model_path)

    metrics = model_params['params']['metrics']
    with open(results_path, 'w') as results:
        results.write('\n'.join(f"{name} = {metric}" for name, metric in zip(['loss'] + metrics, evaluate)))

    if train_params['plot_history']:
        utils.plot_history(history, plot_path)
    
    return ['loss']+metrics, evaluate

def predict(data_path : str, model_path : str, model_params : dict, embeddings_params : dict,
                                                                    preprocess_params : dict):

    model = _create_model(None, True, model_path, model_params)
    input_shape = model._input_shape
    
    sentences_with_embeddings, _ = _data_preprocess(data_path, False, embeddings_params,
                                                      preprocess_params, input_shape[0])
    
    predictions = pd.Series(model.predict(sentences_with_embeddings).ravel())

    save_path = 'data_output/'
    save_path, count = utils.savefile_name(save_path, "Result")
    csv_path = save_path / f"data.csv"
    predictions.to_csv(csv_path, index = False)

def _data_preprocess(data_path : str, train : bool, embeddings_params : dict, preprocess_params : dict,
                                                                  padding_length : int = None) -> list:
    sentences, labels = utils.load_sentences(data_path, train)
    preprocess_params['mode'] = embeddings_params['type']
    preprocess = partial(utils.preprocess,**preprocess_params)
    sentences = sentences.apply(preprocess)

    if embeddings_params["type"] == "word":
        model = utils.load_word_embeddings(embeddings_params['path'],
                                           embeddings_params['binary'],
                                           embeddings_params['convert_to_w2v'])
        sentences_with_embeddings = utils.get_word_embeddings(sentences, model,
                                                              embeddings_params['length'],
                                                              padding_length)
    elif embeddings_params["type"] == "sentence":
        model = utils.load_USE_embeddings(embeddings_params['path'])
        sentences_with_embeddings = utils.get_sentence_embeddings(sentences, model)
    else:
        raise ValueError("Embedding type must be 'word' or 'sentence'.")
    
    return sentences_with_embeddings, labels

def _create_model(input_shape : tuple, load_model : bool, model_path : str, model_params : dict):
    model = models_dict[model_params['type']]
    model = model.load(model_path) if load_model else model(input_shape,**model_params['params'])
    return model
