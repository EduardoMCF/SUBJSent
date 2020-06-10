from subj_sent import train, predict
from pathlib import Path
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--task', action='store', choices=['train', 'predict'],
                    required = True, help = "The task that will be executed.")

args = parser.parse_args()

path = Path(__file__).parent / 'config.json'
with path.open('r') as json_config:
    config = json.load(json_config)

data_path = config['data_path']
model_path = config['model_path']
train_params = config['train']
model_type = config['model']['type']
model_params = {'type' : model_type, 'params' : config['model'][model_type]}
preprocess_params = config['preprocess']
embedding_type = config['embeddings']['type']
embeddings_params = {'type' : embedding_type, **config['embeddings'][embedding_type]}


if args.task == "train":
    metric_names, metrics = train(data_path, model_path, model_params,
                            embeddings_params, train_params, preprocess_params)
    
    print('\n'.join(f"{name} = {metric}" for name, metric in zip(metric_names, metrics)))
else:
    predict(data_path, model_path, model_params, embeddings_params, preprocess_params)
    









