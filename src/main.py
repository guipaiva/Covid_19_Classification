import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.definitions import DATA_DIR, LOGS_DIR
from trainers.simple_trainer import SimpleTrainer
from models import densenet121, resnet50, resnet50V2, vgg16, xception
from data_loaders import covidx_loader
import utils.tools as tools
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import numpy as np



if __name__ == "__main__":
    dataset_dir = os.path.join(DATA_DIR, 'COVIDx_binary')
    train_dir = os.path.join(dataset_dir, 'train')
    label_dir = os.path.join(dataset_dir, 'train.txt')
    im_shape = (224, 224, 3)
    loader = covidx_loader.CovidxLoader(
        train_dir, im_shape, label_dir, 'binary')

    train_data = loader.get_train_ds()
    validation_data = loader.get_validation_ds()
    data = {'train': train_data, 'validation': validation_data}

    model = resnet50V2.ResNet50V2(im_shape, True, 1)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(
        data['train'].classes), y=data['train'].classes)

    weights = dict(enumerate(class_weights))
    
    print(weights)

    trainer = SimpleTrainer(model.model, model.name,
                            data, epochs=12, class_weight=weights)

    history = trainer.train()
    tools.write_raw(history.history, LOGS_DIR, model.name)
