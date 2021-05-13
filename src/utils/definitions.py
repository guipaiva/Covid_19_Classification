import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

MODELS_DIR = os.path.join(SRC_DIR, 'models')
TRAINERS_DIR = os.path.join(SRC_DIR, 'trainers')
LOADERS_DIR = os.path.join(SRC_DIR, 'data_loader')

MODELS_SHAPE = {
    'Xception': {
        'min': (71,71,3),
        'default': (299,299,3)
    },
    'DenseNet121': {
        'min': (32,32,3),
        'default': (224,224,3)
    },
    'ResNet50': {
        'min': (32,32,3),
        'default': (224,224,3)
    },
    'ResNet50V2': {
        'min': (32,32,3),
        'default': (224,224,3)
    },
    'VGG16': {
        'min': (32,32,3),
        'default': (224,224,3)
    }
}