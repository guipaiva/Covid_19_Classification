from utils.definitions import MODELS_SHAPE

class BaseLoader(object):
	def __init__(self, directory, im_shape, batch_size):
		self.im_shape = im_shape
		self.batch_size = batch_size
		self.directory = directory

	def check_shape(self):
		print('Data Loader - Checking image size constraints...')
		try:
			req_dimension = MODELS_SHAPE[self.name]
			assert len(self.im_shape) == 3 and (self.im_shape >= req_dimension['min'])
		except AssertionError:
				print('Error: Image required to have 3 channels and with and height should not be smaller than', req_dimension['min'][0])
		print('Data Loader - Image size check successful')

	def get_train_ds(self):
		raise NotImplementedError

	def get_validation_ds(self):
		raise NotImplementedError
