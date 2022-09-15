import os.path as osp
import os
import logging
from collections import defaultdict
import random

'''
This is a Clothes Changing Dataset Published in ACCV 2020
Evaluation Settings:
    use both cloth-consistent and cloth-changing samples in train set for training
    for testing, including:
    -Standard Setting:
        like standard dataset evaluation, the images in the test set with the same identity and the same camera view
        are discarded when computing evaluation scores
    -Cloth-changing Setting:
        the images with same identity, camera view and clothes are discarded during testing
'''
class LTCC(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir,'train')
        test_dir = os.path.join(self.dataset_dir,'test')
        query_dir = os.path.join(self.dataset_dir,'query')

        self.train, num_train_pids, num_train_imgs = self._process_data(train_dir,relabel=True)
        self.test, num_test_pids, num_test_imgs = self._process_data(test_dir)
        self.query, num_query_pids, num_query_imgs = self._process_data(query_dir)

        logging.info("=> LTCC ReID loaded")
        logging.info("Dataset statistics:")
        logging.info("  ------------------------------")
        logging.info("  subset   | # ids | # imgs | # Avg Clothes Num")
        logging.info("  ------------------------------")
        logging.info("  train    | {:5d} |{:8d}| ".format(num_train_pids, num_train_imgs))
        logging.info("  query    | {:5d} |{:8d}| ".format(num_query_pids, num_query_imgs))
        logging.info("  gallery  | {:5d} |{:8d}| ".format(num_test_pids, num_test_imgs))
        logging.info("  ------------------------------")

        self.num_train_pids = num_train_pids
        self.train_sample = random.sample(self.train, 32)


    def _process_data(self,dir,relabel=False):
        data_list = []
        clothes_ids = defaultdict(set)
        files = os.listdir(dir)
        data_dict = defaultdict(list)
        for file in files:
            if not '.png' in file:
                continue
            # parse the inf data
            img_path = os.path.join(dir,file)
            p_id = int(file.split('_')[0])
            clothes_id = int(file.split('_')[1])
            camera_id = int(file.split('_')[2].split('c')[1]) - 1
            mask_path = os.path.join(dir+'-mask',file)

            data_dict[p_id].append((img_path, camera_id, clothes_id, mask_path))
            clothes_ids[p_id].add(clothes_id)

        id_set = set(data_dict.keys())
        data_list_output = []

        # relabel id to continues
        id_list = list(id_set)
        if relabel:
            id_list.sort()
            pre_clothes_num = 0
            for id_relabel, id_ in enumerate(id_list):
                for data in data_dict[id_]:
                    clothes_id = data[2]
                    clothes_id_pos = list(set(clothes_ids[id_])).index(clothes_id)
                    # cal unique_clothes_id
                    unique_clothes_id = pre_clothes_num + clothes_id_pos
                    data_list_output.append({
                        'img_path': data[0],
                        'pid': id_relabel,
                        'cid': data[1],
                        'clothes_id': data[2],
                        'mask_path': data[3],
                        'unique_clothes_id': unique_clothes_id
                    })
                pre_clothes_num += len(set(clothes_ids[id_]))
        else:
            for id_ in id_list:
                for data in data_dict[id_]:
                    data_list_output.append({
                        'img_path': data[0],
                        'pid': id_,
                        'cid': data[1],
                        'clothes_id': data[2],
                        'mask_path': data[3],
                        'unique_clothes_id': -1
                    })

        return data_list_output, len(id_set), len(data_list_output)
