import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Image_generator:
	im_directory = 'C:\\Users\\X1200WK_05\\Documents\\Guilherme\\COVID-19 Radiography Database'
	def __init__(self, im_size, data_augmentation = False, batch_size = 64):
		self.valid_generator = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
		self.im_size = im_size
		self.batch_size = batch_size
		self.data_augmentation = data_augmentation
		if(data_augmentation):
			self.train_generator = ImageDataGenerator(
				rescale = 1./255,
				validation_split = 0.2,
				rotation_range=20,
				width_shift_range=0.2,
				height_shift_range=0.2,
				horizontal_flip=True
			)
		else:
			self.train_generator =  ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

	def generate(self):
		train_ds = self.train_generator.flow_from_directory(
			#save_to_dir= r"C:\Users\X1200WK_05\Documents\Guilherme\NNs\Models\train",
			directory = self.im_directory,
			seed = 36,
			class_mode = 'categorical',
			batch_size = self.batch_size,
			target_size = (self.im_size,self.im_size),
			subset = 'training'
		)
		
		vald_ds = self.valid_generator.flow_from_directory(
			#save_to_dir= r"C:\Users\X1200WK_05\Documents\Guilherme\NNs\Models\valid",
			directory = self.im_directory,
			seed = 36,
			class_mode = 'categorical',
			batch_size = self.batch_size,
			target_size = (self.im_size,self.im_size),
			subset = 'validation'
		)

		return train_ds, vald_ds

'''
td1, vd1 = Image_generator(224, data_augmentation=False).generate()
print('Gerando')
#td2, vd2 = Image_generator(224, data_augmentation=True).generate()
count = 0
print('Entrando')
for image, label in td1:
	count += 1
	print(count)
	if(count > 2500):
		break'''
