from trainers.simple_trainer import SimpleTrainer
from models import vgg16, xception, densenet121, resnet50V2, simple_cnn
from utils.definitions import DATA_DIR
from data_loaders import CovidxLoader, SirmLoader
import os

if __name__ == "__main__":
	dataset = 'COVIDx'
	im_directory = os.path.join(DATA_DIR, dataset)
	label_dir = os.path.join(DATA_DIR)
	im_shape = (299,299,3)
	loader = data_loader.CovidxLoader(im_directory, im_shape)

	train_data = loader.get_train_ds()
	validation_data = data_loader.get_validation_ds()
	data = {'train': train_data, 'validation': validation_data}

	model = simple_cnn.SimpleCNN(im_shape)
	
	trainer = SimpleTrainer(model.model, model.name, data, epochs = 12)

	trainer.train()
