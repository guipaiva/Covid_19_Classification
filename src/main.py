import numpy as np
import utils.tools as tools
from data_loaders import covidx_loader
from models import densenet121, resnet50, resnet50V2, vgg16, xception
from trainers import binary_trainer, multiclass_trainer
from utils.definitions import DATA_DIR, LOGS_DIR
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    models = [densenet121.DenseNet121, resnet50V2.ResNet50V2,
              resnet50.ResNet50, vgg16.VGG16, xception.Xception]

    dataset_dir = os.path.join(DATA_DIR, 'COVIDx_Multiclass')
    train_dir = os.path.join(dataset_dir, 'train')
    label_dir = os.path.join(dataset_dir, 'train.txt')
    im_shape = (224, 224, 3)
    loader = covidx_loader.CovidxLoader(
        train_dir, im_shape, label_dir, 'categorical')

    train_data = loader.get_train_ds()
    print(train_data.class_indices)
    validation_data = loader.get_validation_ds()
    data = {'train': train_data, 'validation': validation_data}

    #TODO: make train.py

    for model in models:
        model = model(im_shape, True, 1)
        trainer = multiclass_trainer.MulticlassTrainer(model.model, model.name, data, epochs=12)
        history = trainer.train()
        tools.write_raw(history.history, LOGS_DIR, model.name)
        print(model.name + ' Trained\n\n')
        K.clear_session()
