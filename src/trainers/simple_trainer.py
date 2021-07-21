from base.base_trainer import BaseTrainer
import datetime
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SimpleTrainer(BaseTrainer):
	def __init__(self, model, name, data, epochs):
		super(SimpleTrainer, self).__init__(model, name, data, epochs)
		self.log_dir = "../TensorBoard/fit/" + self.name + "/" + datetime.datetime.now().strftime("%d%m")
		self.accuracy = []
		self.loss = []
		self.val_accuracy = []
		self.val_loss = []
		self.class_weights = {
			0: (1/self.data["train"])
		}
		

	def train(self):
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
		history = self.model.fit(
			self.data["train"],
			validation_data = self.data["validation"],
			callbacks = [tensorboard_callback],
			epochs = self.epochs,
			verbose = 1
		)
		self.loss.extend(history.history['loss'])
		self.accuracy.extend(history.history['accuracy'])
		self.val_loss.extend(history.history['val_loss'])
		self.val_accuracy.extend(history.history['val_accuracy'])
		