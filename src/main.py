from data_loader.simple_loader import SimpleLoader
from trainers.simple_trainer import SimpleTrainer
from models import vgg16

if __name__ == "__main__":
	im_directory = "../data/COVID-19 Radiography Database"
	im_shape = (224,224,3)
	data_loader = SimpleLoader(im_directory, im_shape)

	train_data = data_loader.get_train_ds()
	validation_data = data_loader.get_validation_ds()
	data = {'train': train_data, 'validation': validation_data}

	model = vgg16.transfer_learning_VGG16(im_shape)

	trainer = SimpleTrainer(model.model, data, epochs = 12)

	trainer.train()