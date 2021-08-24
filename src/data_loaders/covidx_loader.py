from tensorflow.keras.preprocessing.image import ImageDataGenerator
from base.base_loader import BaseLoader
import pandas as pd
from sklearn.model_selection import train_test_split


class CovidxLoader(BaseLoader):
    def __init__(self, directory, im_shape, label_dir, class_mode, transfer_learn, batch_size=32):
        super(CovidxLoader, self).__init__(directory, im_shape, batch_size)
        if transfer_learn:
            self.generator = ImageDataGenerator()
        else:
            self.generator = ImageDataGenerator(rescale=1./255)
            
        self.dataframe = pd.read_csv(
            label_dir,
            sep=' ',
            names=['id', 'filename', 'label', 'source'],
            usecols=['filename', 'label'],
            header=None
        )

        self.train_df, self.valid_df = train_test_split(
            self.dataframe,
            test_size=0.2,
            random_state=36,
            stratify=self.dataframe['label']
        )

        self.class_mode = class_mode
        print('='*20, '\n', 'Train:')
        print(self.train_df['label'].value_counts())
        print('='*20, '\n', 'Valid:')
        print(self.valid_df['label'].value_counts())
        #print(self.dataframe.loc[self.dataframe['label'] == 'positive'])

    def get_train_ds(self):
        train_ds = self.generator.flow_from_dataframe(
            dataframe=self.train_df,
            directory=self.directory,
            x_col='filename',
            y_col='label',
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
            seed=12,
            target_size=(self.im_shape[:-1])
        )
        return train_ds

    def get_validation_ds(self):
        validation_ds = self.generator.flow_from_dataframe(
            dataframe=self.valid_df,
            directory=self.directory,
            x_col='filename',
            y_col='label',
            class_mode=self.class_mode,
            batch_size=self.batch_size,
            shuffle=True,
            seed=12,
            target_size=self.im_shape[:-1]
        )

        return validation_ds
