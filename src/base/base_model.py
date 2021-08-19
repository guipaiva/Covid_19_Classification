import tensorflow as tf
#pylint: disable=import-error
from utils.definitions import MODELS_SHAPE


class BaseModel(object):
    def __init__(self, name, im_shape, transfer_learn, classes, metrics):
        self.name = name
        self.im_shape = im_shape
        self.transfer_learn = transfer_learn
        self.classes = classes
        self.metrics = metrics

        self.activation_function = 'sigmoid' if classes == 1 else 'softmax'
        self.weights = 'imagenet' if self.transfer_learn else None
        self.layers_trainable = False if self.transfer_learn else True
        self.loss = 'binary_crossentropy' if classes == 1 else 'categorical_crossentropy'

    def build_model(self):
        raise NotImplementedError
