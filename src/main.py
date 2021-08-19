import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

import trainer
import utils.tools as tools
from data_loaders import covidx_loader
from models import densenet121, resnet50, resnet50V2, vgg16, xception
from utils.definitions import DATA_DIR, LOGS_DIR

if __name__ == "__main__":
    CLASS_MODE = 'multiclass'

    models = [densenet121.DenseNet121, resnet50V2.ResNet50V2,
              resnet50.ResNet50, vgg16.VGG16, xception.Xception]



    dataset_dir = os.path.join(DATA_DIR, 'COVIDx_Multiclass')
    train_dir = os.path.join(dataset_dir, 'train')
    label_dir = os.path.join(dataset_dir, 'train.txt')
    im_shape = (224, 224, 3)

    loader = covidx_loader.CovidxLoader(directory=train_dir,
                                        im_shape=im_shape,
                                        label_dir=label_dir,
                                        class_mode='categorical')

    train_data = loader.get_train_ds()
    # print(train_data.class_indices)
    validation_data = loader.get_validation_ds()
    data = {'train': train_data, 'validation': validation_data}

    # TODO: make train.py

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    if CLASS_MODE == 'binary':
        METRICS.append(keras.metrics.BinaryAccuracy())
    else:
        METRICS.append(keras.metrics.CategoricalAccuracy())

    for model in models:
        model = model(im_shape=im_shape, transfer_learn=True,
                      classes=3, metrics=METRICS)

        trainer = trainer.Trainer(
            model= model.model,
            name = model.name,
            data = data,
            epochs= 12,
            class_mode= CLASS_MODE
        )
        history = trainer.train()

        logs_dir = os.path.join(LOGS_DIR, *['raw', CLASS_MODE])

        tools.write_raw(history.history, log_dir=logs_dir,
                        model_name=model.name)

        print(model.name + ' Trained\n\n')
        K.clear_session()
