# import pandas as pd
import json
import logging
import math

import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split


class Predictor:
    def __init__(self):
        self.init_model()
        self._data_file = open('dataset/dataset.txt', "a")
        self._data = pd.read_csv('dataset/dataset.txt', header=None, names=['job_id', 'conf_id', 'cost', 'timestamp'])
        self._config_info = json.load(open("worker_conf.json"))
        self._cpu_rate = 0.0000001
        self._memory_rate = 0.000000001
        self._max_perf_value = 300102.0
        self._min_perf_value = 1.3333333333333333
        self._configs = self._config_info.keys()
        self._data = self._data[self._data.conf_id.isin(self._configs)]
        self._trainset = None
        self._testset = None
        self._reader = Reader(rating_scale=(0, 1))
        self._dataset = Dataset(self._reader)
        self.log = logging.getLogger('Predictor')

    def init_model(self):
        self._model = SVD(n_factors=15, lr_all=.01, reg_bu=0, reg_bi=0, reg_pu=.5, reg_qi=0.05, n_epochs=10)

    def write_to_file(self, data):
        self._data_file.write(",".join([str(item) for item in data]) + '\n')
        self._data_file.flush()

    def get_objective_value(self, conf_id, norm_val, objective):
        runtime = self.denormalize(norm_val)
        if objective == "perf":
            cost = runtime
            return cost
        if objective == "cost":
            cpu, memory = [self._config_info[str(conf_id).zfill(2)].get(key) for key in ['cpus', 'memory']]
            cost = (cpu * self._cpu_rate + memory * self._memory_rate) * runtime
            return cost
        if objective == "cost+perf":
            cpu, memory = [self._config_info[str(conf_id).zfill(2)].get(key) for key in ['cpus', 'memory']]
            cost = (cpu * self._cpu_rate + memory * self._memory_rate) * (runtime ** 2)
            return cost

    def normalize(self, perf_value):
        return (perf_value - self._min_perf_value) / (self._max_perf_value - self._min_perf_value)

    def denormalize(self, norm_value):
        perf_value = (norm_value * (self._max_perf_value - self._min_perf_value)) + self._min_perf_value
        return perf_value

    def get_avg_difference(self, v_2, v_1):
        value = abs(v_2 - v_1) / ((v_2 + v_1) / 2)
        return value

    def add_data_node(self, node: list, save=True):
        self.log.info(f"SVD Got Node:{node}")
        node[2] = self.normalize(node[2])
        # --------------------------#
        existing_data = self._data[(self._data['job_id'] == node[0]) & (self._data['conf_id'] == node[1])]
        if len(existing_data) > 0:
            if len(existing_data) >= 3:
                return
            avg_diff = self.get_avg_difference(existing_data['cost'].mean(), node[2])
            if avg_diff > .1:
                return
        # --------------------------#
        if save:
            self.write_to_file(node)
        node.append(None)
        self.log.info(f"Addind Node:{node}")
        self._data.loc[len(self._data)] = node

    def train(self):
        data = self._data.groupby(['job_id', 'conf_id', 'timestamp'])['cost'].mean().reset_index()
        data = data[['job_id', 'conf_id', 'cost', 'timestamp']]
        self._trainset = self._dataset.construct_trainset(raw_trainset=data.values)
        self._testset = self._trainset.build_anti_testset()
        self.init_model()
        # print("Train set size:", len(list(self._trainset.all_ratings())))
        self._model.fit(trainset=self._trainset)

    def construct_test_set_for_job(self, job_id: int):
        train_conf_ids = [int(x[1]) for x in self._data[self._data['job_id'] == job_id].values]
        test_set = []
        for config in self._configs:
            if int(config) not in train_conf_ids:
                test_set.append([job_id, int(config), None, None])

        return self._dataset.construct_testset(raw_testset=test_set)

    def test_configs_for_job(self, job_id: int):
        test_set = self.construct_test_set_for_job(job_id)
        pred_set = []
        for x in test_set:
            pred = self._model.predict(x[0], x[1], x[2])
            pred_set.append((pred.uid, pred.iid, pred.est))
        return pred_set

    def recommend_config_for_job(self, job_id: int, objective):
        job_data_set = self._data[self._data['job_id'] == job_id].values
        pred_data_set = self.test_configs_for_job(job_id)

        train_best = min(job_data_set, key=lambda x: self.get_objective_value(int(x[1]), int(x[2]), objective))
        pred_best = min(pred_data_set, key=lambda x: self.get_objective_value(int(x[1]), int(x[2]), objective))

        return int(min([train_best, pred_best], key=lambda x: x[2])[1])


# if __name__ == "__main__":
#     pred = Predictor()
#
#     nodes = []
#     # nodes = [[2, 1, 3],
#     #          [1, 1, 5],
#     #          [1, 8, 2],
#     #          [2, 8, 5]]
#     with open("dataset/dataset.txt", "r") as file:
#         for line in file:
#             vals = line.split(',')[:-1]
#             nodes.append([int(float(vals[0])), int(float(vals[1])), float(vals[2])])
#
#     for node in nodes:
#         # print(type(node))
#         pred.add_data_node(node, False)
#
#     pred.train()
#     # print(pred._model.pu.shape)
#     # print()
#     # print(pred._model.qi.shape)
#     guess = pred.recommend_config_for_job(8, "cost")
#     print(guess)
#     # print([x.est for x in guess])

# print(float("30 milliseconds".split()[0]))
