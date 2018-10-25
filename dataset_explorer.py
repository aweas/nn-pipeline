import numpy as np
import pandas as pd


class DatasetExplorer:
    def __init__(self, filename):
        self.X = None
        self.y = None
        self.filename = filename
        self.directory = '/'.join(filename.split('\\')[:-1])+'/recordings'

    def _read_dataset_from_file(self):
        dataset = pd.read_csv(self.filename.replace('\\', '/'))
        dataset = dataset[dataset['conflict'] == False]
        self.X = pd.Series(self.directory + '/' + dataset['uuid'] + '.bmp')
        self.X = self.X.values
        self.y = dataset['model'].values

    def reset(self):
        self.prepare()

    def prepare(self):
        self._read_dataset_from_file()
        return self

    def get_data(self):
        return self.X, self.y