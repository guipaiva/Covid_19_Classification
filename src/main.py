from data_loader.simple_loader import SimpleLoader
from trainers.simple_trainer import SimpleTrainer
from models import vgg16, xception, densenet121, resnet50V2, simple_cnn

if __name__ == "__main__":
	im_directory = "../data/COVID-19 Radiography Database"
	im_shape = (299,299,3)
	data_loader = SimpleLoader(im_directory, im_shape)

	train_data = data_loader.get_train_ds()
	validation_data = data_loader.get_validation_ds()
	data = {'train': train_data, 'validation': validation_data}

	model = simple_cnn.SimpleCNN(im_shape)
	
	trainer = SimpleTrainer(model.model, model.name, data, epochs = 12)

	trainer.train()
