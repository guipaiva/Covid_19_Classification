from collections import Counter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainers.simple_trainer import SimpleTrainer
from models import vgg16, xception, densenet121, resnet50V2, resnet50
from utils.definitions import DATA_DIR
from data_loaders import covidx_loader


if __name__ == "__main__":
	dataset_dir = os.path.join(DATA_DIR, 'COVIDx_binary')
	train_dir = os.path.join(dataset_dir, 'train')
	label_dir = os.path.join(dataset_dir, 'train.txt')
	im_shape = (224,224,3)
	loader = covidx_loader.CovidxLoader(train_dir, im_shape, label_dir, 'binary')

	train_data = loader.get_train_ds()
	validation_data = loader.get_validation_ds()
	data = {'train': train_data, 'validation': validation_data}

	model = resnet50V2.ResNet50V2(im_shape, True, 1)
	counter = Counter(data['train'].classes)
	samples = data['train'].samples
	neg_weight = (1 / counter['0']) * (samples / 2.0)
	pos_weight = (1 / counter['1']) * (samples / 2.0)
	weights = {0: neg_weight, 1: pos_weight}

	trainer = SimpleTrainer(model.model, model.name, data, epochs = 12, class_weight= weights)

	trainer.train()
