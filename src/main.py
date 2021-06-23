from trainers.simple_trainer import SimpleTrainer
from models import vgg16, xception, densenet121, resnet50V2, resnet50
from utils.definitions import DATA_DIR
from data_loaders import covidx_loader
import os

if __name__ == "__main__":
	dataset_dir = os.path.join(DATA_DIR, 'COVIDx')
	train_dir = os.path.join(dataset_dir, 'train')
	label_dir = os.path.join(dataset_dir, 'train.txt')
	im_shape = (299,299,3)
	loader = covidx_loader.CovidxLoader(train_dir, im_shape, label_dir, 'binary')

	train_data = loader.get_train_ds()
	validation_data = loader.get_validation_ds()
	data = {'train': train_data, 'validation': validation_data}

	model = densenet121.DenseNet121(im_shape, 1, False)
	
	trainer = SimpleTrainer(model.model, model.name, data, epochs = 12)

	trainer.train()
