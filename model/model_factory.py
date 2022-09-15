import imp
import logging
from .net.resnet import resnet50
from .net.transreid import TCiP, TransReID

__factory_model = {
    'resnet50': resnet50,
    'transreid': TransReID,
    'TCiP': TCiP
}


def make_model(model_name, model_params: dict):
    if not model_name in __factory_model.keys():
        logging.error(f'model : [{model_name}] is not defined')
    return __factory_model[model_name](**model_params)
