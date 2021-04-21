class BaseLoader(object):
	def __init__(self, directory, im_shape, batch_size, label_dir = None):
		self.im_shape = im_shape
		self.batch_size = batch_size
		self.directory = directory

	def get_train_ds(self):
		raise NotImplementedError

	def get_validation_ds(self):
		raise NotImplementedError
