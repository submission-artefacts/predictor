# import pandas as pd
import json
import math
import numpy as np
import pandas as pd

from als_recommendation import ALS
import logging


class Predictor:
    def __init__(self):
        self._data_file = open('dataset/dataset.txt', "a")
        self._data = pd.read_csv('dataset/dataset.txt', header=None, names=['job_id', 'conf_id', 'cost'])
        self._config_info = json.load(open("worker_conf.json"))
        self._cpu_rate = 0.0000001
        self._memory_rate = 0.000000001
        self._max_perf_value = 300102.0
        self._min_perf_value = 1.3333333333333333
        self._configs = list(self._config_info.keys())
        self._data = self._data[self._data.conf_id.isin([int(x) for x in self._configs])]
        self._none_val = 99
        self._matrix = np.array(self._data.pivot(index='job_id', columns='conf_id', values='cost').fillna(self._none_val))
        self._model: ALS = None
        self.log = logging.getLogger('Predictor')

    def init_model(self):
        self._model = ALS(X=self._matrix, K=15, lamb_u=.05, lamb_v=.0005, max_epoch=10, none_val=self._none_val)

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

    def update_matrix(self):
        data = self._data.groupby(['job_id', 'conf_id'])['cost'].mean().reset_index()
        self._matrix = np.array(data.pivot(index='job_id', columns='conf_id', values='cost').fillna(self._none_val))

    def get_avg_difference(self, v_2, v_1):
        v_2 = self.denormalize(v_2)
        v_1 = self.denormalize(v_1)
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
        self.log.info(f"Addind Node:{node}")
        self._data.loc[len(self._data)] = node
        if save:
            self.write_to_file(node)

    def train(self):
        if len(self._data)!=0:
            self.update_matrix()
            self.init_model()
            self._model.train()

    def recommend_config_for_job(self, job_id: int, objective):
        est_perf_values = self._model.get_est_matrix()[job_id - 1]
        self.log.info(f"Estimations for job: {job_id}")
        self.log.info(est_perf_values)
        best_config = None
        best_cost = self._none_val
        for conf_idx, est_value in enumerate(est_perf_values):
            config = self._configs[conf_idx]
            cost = self.get_objective_value(config, est_value, objective)
            if cost < best_cost:
                best_config = config
                best_cost = cost
        return int(best_config)

#     def save_est_matrix(self):
#         est_matrix = self._model.get_est_matrix()
#         self.log.info(est_matrix)
#         np.savetxt(self._est_matrix_file_path, est_matrix, delimiter=',', fmt="%s")
#
#
# def reset_rating_matrix_file():
#     a = np.full((2, 16), 99)
#     np.savetxt("dataset/rating_matrix.csv", a, delimiter=',', fmt="%s")
#
#
# if __name__ == "__main__":
#     reset_rating_matrix_file()
