"""
download dataset from https://www.kaggle.com/datasets/kazanova/sentiment140 and unzip before loading

go to https://www.kaggle.com/settings to get kaggle token
"""
import os
from typing import Union

from kaggle.api.kaggle_api_extended import KaggleApi
import csv
import copy
import pandas as pd

_dataset: dict[str, list] = {
    'target': [],
    'ids': [],
    'date': [],
    'user': [],
    'text': []
}

def load_dataset(path: str='./dataset', return_dict: bool=False) -> Union[dict, pd.DataFrame]:
    d = copy.deepcopy(_dataset)

    path = os.path.join(path, 'training.1600000.processed.noemoticon.csv')

    if not os.path.exists(path):
        download(path)

    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        for target, ids, date, flag, user, text in csv.reader(f):
            d['target'].append(int(target))
            d['ids'].append(int(ids))
            d['date'].append(date if len(date) else None)
            d['user'].append(user if len(user) else None)
            d['text'].append(text if len(text) else None)

    if return_dict:
        return d
    else:
        return pd.DataFrame(d)

def download(path: str='./dataset'):
    api = KaggleApi()
    api.authenticate()

    os.makedirs(path, exist_ok=True)

    api.dataset_download_files('kazanova/sentiment140', path, quiet=False, unzip=True)

if __name__ == '__main__':
    download()