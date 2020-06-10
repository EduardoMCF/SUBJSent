import matplotlib.pyplot as plt

import os
from pathlib import Path

def log(msg : str):
    def log_decorator(func):
        def wrapper(*args,**kwargs):
            print(f"{msg}...", end = " ")
            result = func(*args,**kwargs)
            print('finished')
            return result
        return wrapper
    return log_decorator

def plot_history(history : object, path : str):
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
    
def savefile_name(path : str, folderName = "Model") -> str:
    path = Path(__file__).parent.parent.parent / path
    files = os.listdir(path)
    count = max(map(lambda name: int(name.split(' ')[-1]),files)) + 1 if files else 1
    path = path / f"{folderName} {count}"
    os.mkdir(path)
    return path,count