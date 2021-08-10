import datetime
import os

import numpy as np
import tensorflow as tf
from base.base_trainer import BaseTrainer
from sklearn.utils.class_weight import compute_class_weight
from utils.definitions import ROOT_DIR


class MulticlassTrainer(BaseTrainer):
    def __init__(self, model, name, data, epochs):
        super(MulticlassTrainer, self).__init__(model, name, data, epochs)
        self.log_dir = os.path.join(ROOT_DIR, 'logs')
        self.model_name = name

        self.class_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(
            data['train'].classes), y=data['train'].classes)))
        print(self.class_weight)

    def train(self):
        tb_folder = "TensorBoard/multiclass/" + self.name + "/" + datetime.datetime.now().strftime("%d%m")
        tb_dir = os.path.join(self.log_dir, tb_folder)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_dir, histogram_freq=1)
        #TODO: Add confusion matrix, F1-Score, ROC AUC

        history = self.model.fit(
            self.data["train"],
            validation_data=self.data["validation"],
            callbacks=[tensorboard_callback],
            class_weight=self.class_weight,
            epochs=self.epochs,
            verbose=1
        )

        return history
