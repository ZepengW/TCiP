'''
this file contains method of reading dataset
'''
from PIL import Image
import logging
import os
from dataset.transforms import build_transforms, build_transforms_shape, build_transformers_value
from dataset.mask import read_person_mask
from torchvision import transforms as T
import torch

class GetImg(object):
    """
    read img data from img path, and resize it to uniform size
    :param data:
    :param resize:
    :return:
    """
    def __init__(self, **kwargs):
        mode = kwargs.get('mode', 'train')
        self.img_size = kwargs.get('image_size')
        self.tf_shape = build_transforms_shape(self.img_size, mode)
        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])
        p = kwargs.get('p', 0.5)
        self.tf_value = build_transformers_value(mean, std, p, mode)
        self.to_tensor = T.ToTensor()

    def __call__(self, data:dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        clothes_id = data.get('clothes_id', -1)
        unique_clothes_id = data.get('unique_clothes_id', -1)
        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            img = None
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(img)   # C H W
            # shape transform
            img = self.tf_shape(img)
            # value transform
            img = self.tf_value(img)
        data_dict = {'img': img, 'pid': pid, 'camera_id': cid, 'img_path': img_path , 'clothes_id':clothes_id, 'unique_clothes_id':unique_clothes_id}
        return data_dict

class GetImgSem(object):
    """
    read img data from img path, and read semantic data from mask_path
    :param data:
    :param resize:
    :return:
    """
    def __init__(self, mode, **kwargs):
        self.img_size = kwargs.get('image_size')
        self.tf_shape = build_transforms_shape(self.img_size, mode)
        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])
        p = kwargs.get('p', 0.5)
        self.tf_value = build_transformers_value(mean, std, p, mode)
        self.to_tensor = T.ToTensor()

    def __call__(self, data:dict):
        img_path = data.get('img_path')
        pid = data.get('pid')
        cid = data.get('cid')
        clothes_id = data.get('clothes_id', -1)
        unique_clothes_id = data.get('unique_clothes_id', -1)
        sem_path = data.get('mask_path')

        if not os.path.isfile(img_path):
            logging.error(f'Can Not Read Image {img_path}')
            raise NotImplementedError
        else:
            img = Image.open(img_path).convert('RGB')
            img = self.to_tensor(img)   # C H W
            sem = torch.tensor(read_person_mask(sem_path))  # S H W
            img_sem = torch.cat([img,sem], dim = 0)
            # shape transform
            img_sem = self.tf_shape(img_sem)
            img = img_sem[0:3]
            sem = img_sem[3:]
            # value transform
            img = self.tf_value(img)
        data_dict = {'img': img,'pid': pid, 'camera_id': cid, 'clothes_id': clothes_id, \
                     'sem': sem ,'img_path': img_path, 'unique_clothes_id':unique_clothes_id}
        return data_dict

