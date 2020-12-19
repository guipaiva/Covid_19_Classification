class BaseTrainer(object):
	def __init__(self, model, name, data, epochs):
		self.model = model
		self.data = data
		self.name = name
		self.epochs = epochs
	def train(self):
		raise NotImplementedError