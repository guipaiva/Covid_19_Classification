import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
from utils.definitions import DATA_DIR, LOGS_DIR
from trainers import binary_trainer, multiclass_trainer
from models import densenet121, resnet50, resnet50V2, vgg16, xception
from data_loaders import covidx_loader
import utils.tools as tools
import numpy as np

if __name__ == "__main__":
    models = [densenet121.DenseNet121, resnet50V2.ResNet50V2,resnet50.ResNet50, vgg16.VGG16, xception.Xception]

    dataset_dir = os.path.join(DATA_DIR, 'COVIDx_Multiclass')
    train_dir = os.path.join(dataset_dir, 'train')
    label_dir = os.path.join(dataset_dir, 'train.txt')
    im_shape = (224, 224, 3)

    loader = covidx_loader.CovidxLoader(directory=train_dir,
                                        im_shape=im_shape,
                                        label_dir=label_dir,
                                        class_mode='categorical')

    train_data = loader.get_train_ds()
    print(train_data.class_indices)
    validation_data = loader.get_validation_ds()
    data = {'train': train_data, 'validation': validation_data}

    # TODO: make train.py
    # model = densenet121.DenseNet121(im_shape=im_shape, transfer_learn=True, classes=3)
    # trainer = multiclass_trainer.MulticlassTrainer(model.model, model.name, data, epochs=12)
    # history = trainer.train()
    # logs_dir = os.path.join(LOGS_DIR, *['raw', 'multiclass'])
    # tools.write_raw(history.history, LOGS_DIR, model.name)
    # print(model.name + ' Trained\n\n')
    
    for model in models:
        model = model(im_shape=im_shape, transfer_learn=True, classes=3)
        trainer = multiclass_trainer.MulticlassTrainer(
            model.model, model.name, data, epochs=12)
        history = trainer.train()
        logs_dir = os.path.join(LOGS_DIR, *['raw', 'multiclass'])
        tools.write_raw(history.history, LOGS_DIR, model.name)
        print(model.name + ' Trained\n\n')
        K.clear_session()
