from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base_loader import BaseLoader


class SimpleLoader(BaseLoader):

	def __init__(self, directory, im_size, batch_size=32):
		super(SimpleLoader, self).__init__(directory, im_size, batch_size=32)
		self.generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

	def get_train_ds(self):
		train_ds = self.generator.flow_from_directory(
			directory=self.directory,
			#seed = 36,
			class_mode='categorical',
			batch_size=self.batch_size,
			target_size=(self.im_size, self.im_size),
			subset='training'
		)

		return train_ds

	def get_validation_ds(self):
		vald_ds = self.generator.flow_from_directory(
			directory=self.directory,
			#seed = 36,
			class_mode='categorical',
			batch_size=self.batch_size,
			target_size=(self.im_size, self.im_size),
			subset='validation'
		)

		return vald_ds

