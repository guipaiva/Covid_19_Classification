from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base.base_loader import BaseLoader


class SirmLoader(BaseLoader):

    def __init__(self, directory, im_shape, batch_size=32):
        super(SirmLoader, self).__init__(directory, im_shape, batch_size)
        self.generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    def get_train_ds(self):
        train_ds = self.generator.flow_from_directory(
            directory=self.directory,
            class_mode='categorical',
            batch_size=self.batch_size,
            target_size=self.im_shape[:-1],
            subset='training'
        )

        return train_ds

    def get_validation_ds(self):
        vald_ds = self.generator.flow_from_directory(
            directory=self.directory,
            class_mode='categorical',
            batch_size=self.batch_size,
            target_size=self.im_shape[:-1],
            subset='validation'
        )

        return vald_ds
