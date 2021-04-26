from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base.base_loader import BaseLoader
import pandas as pd

class CovidxLoader(BaseLoader):
	def __init__(self, directory, label_dir, im_shape, batch_size=32):
		super(CovidxLoader, self).__init__(directory, im_shape, batch_size, label_dir)
		self.generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
		self.dataframe = pd.read_csv(label_dir)

	def get_train_ds(self):
		train_ds = self.generator.flow_from_dataframe(
			dataframe=self.dataframe,
			directory=self.directory,
			x_col='image',
			y_col='label',
			class_mode='binary',
			batch_size=self.batch_size,
			seed=36,
			subset='training',
			target_size=self.im_shape[:-1]
		)

		return train_ds

	def get_validation_ds(self):
		train_ds = self.generator.flow_from_dataframe(
			dataframe=self.dataframe,
			directory=self.directory,
			x_col='image',
			y_col='label',
			class_mode='binary',
			batch_size=self.batch_size,
			seed=36,
			subset='validation',
			target_size=self.im_shape[:-1]
		)

		return validation_ds
	