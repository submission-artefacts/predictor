# import pandas as pd
import logging

import pandas as pd


class Collector:
    def __init__(self):
        self._data_file = open('dataset/dataset.txt', "a")
        self.log = logging.getLogger('Collector')

    def write_to_file(self, data):
        self._data_file.write(",".join([str(item) for item in data]) + '\n')
        self._data_file.flush()


    def add_data_node(self, node: list):
        self.write_to_file(node[:3])

