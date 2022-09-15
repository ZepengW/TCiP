from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid, _, _).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, data_dict in enumerate(self.data_source):
            pid = data_dict.get('pid')          # Need to forcibly specify pid as the second data
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length



class RandomIdentitySamplerCC(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances with C different clothes num
    therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid, _, _).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_per_clothes):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_per_clothes = num_per_clothes
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(dict) #dict with dict value
        self.pid_num = defaultdict(int)
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, data_dict in enumerate(self.data_source):
            pid = data_dict.get('pid')          # Need to forcibly specify pid as the second data
            clothes_id = data_dict.get('clothes_id')
            if clothes_id in self.index_dic[pid].keys():
                self.index_dic[pid][clothes_id].append(index)
            else:
                self.index_dic[pid][clothes_id] = [index]
            self.pid_num[pid] += 1
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            num = self.pid_num[pid]
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            pid_dict = copy.deepcopy(self.index_dic[pid])
            clothes_ids = list(pid_dict.keys())
            flag_get = True
            idxs = []
            while(flag_get):
                random.shuffle(clothes_ids)
                batch_idxs = []
                for clothes_id in clothes_ids:
                    clothes_id_idx = pid_dict[clothes_id]
                    if len(clothes_id_idx) <= 1:
                        flag_get = False
                        continue
                    elif len(clothes_id_idx) < self.num_per_clothes:
                        selected_clothes_id_idx = np.random.choice(clothes_id_idx, size=self.num_instances, replace=True)
                        # flag_get = False
                        # continue
                    else:
                        random.shuffle(clothes_id_idx)
                        selected_clothes_id_idx = clothes_id_idx[0:self.num_per_clothes]
                    selected_clothes_id_idx = list(selected_clothes_id_idx)
                    flag_get = True
                    pid_dict[clothes_id] = list(set(clothes_id_idx) - set(selected_clothes_id_idx))
                    idxs += selected_clothes_id_idx
                    break
            batch_idxs = []
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length