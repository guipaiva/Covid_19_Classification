import csv
import datetime
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
# pylint: disable=import-error
from utils.definitions import LOGS_DIR


class Trainer:
    def __init__(self, model, name, data, epochs, class_mode):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.model_name = name
        self.class_mode = class_mode

        n_classes = np.unique(data['train'].classes)
        labels = data['train'].classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=n_classes,
            y=labels
        )
        self.class_weight = dict(enumerate(class_weights))

    def train(self):
        tb_folder = 'TensorBoard' + '/' + self.class_mode + "/" + self.model_name + \
            "/" + datetime.datetime.now().strftime("%d%m")
        tb_dir = os.path.join(LOGS_DIR, tb_folder)
        csv_folder = 'raw' + '/' + self.class_mode
        csv_dir = os.path.join(LOGS_DIR, *[csv_folder, f'{self.model_name}.csv'])
        os.makedirs(csv_dir, exist_ok = True)
        csv_logger = CSVLogger(csv_dir + '/train.csv')
        tensorboard_callback = TensorBoard(log_dir=tb_dir, histogram_freq=1)
        # TODO: Add confusion matrix, F1-Score, ROC AUC

        history = self.model.fit(
            self.data["train"],
            validation_data=self.data["validation"],
            callbacks=[tensorboard_callback, csv_logger],
            class_weight=self.class_weight,
            epochs=self.epochs,
            verbose=1
        )

        return history
