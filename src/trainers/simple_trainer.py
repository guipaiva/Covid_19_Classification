from base.base_trainer import BaseTrainer
import datetime
import tensorflow as tf
from utils.definitions import ROOT_DIR
import os
import csv

class SimpleTrainer(BaseTrainer):
	def __init__(self, model, name, data, epochs, class_weight):
		super(SimpleTrainer, self).__init__(model, name, data, epochs)
		self.log_dir = os.path.join(ROOT_DIR,'logs')
		self.model_name = name
		self.class_weight = class_weight

	def train(self):
		tb_folder = "TensorBoard/fit/" + self.name + "/" + datetime.datetime.now().strftime("%d%m")
		tb_dir = os.path.join(self.log_dir, tb_folder)
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
		history = self.model.fit(
			self.data["train"],
			validation_data = self.data["validation"],
			callbacks = [tensorboard_callback],
			class_weight = self.class_weight,
			epochs = self.epochs,
			verbose = 1
		)

		return history
	
	


		

		