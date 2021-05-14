from utils.definitions import MODELS_SHAPE

class BaseModel(object):
	def __init__(self, name, im_shape, transfer_learn):
		self.model = None
		self.name = name
		self.im_shape = im_shape
		self.transfer_learn = transfer_learn

	def check_shape(self):
		print('Checking image size constraints...')
		try:
			req_dimension = MODELS_SHAPE[self.name]
			if self.transfer_learn:
				assert len(self.im_shape) == 3 and (self.im_shape >= req_dimension['min'])
			else:
				assert self.im_shape == req_dimension['default']
		except AssertionError:
			if self.transfer_learn:
				print('Error: Image required to have 3 channels and with and height should not be smaller than', req_dimension['min'][0])
			else:
				print('Error: Image shape without transfer learning must be', end = ' ')
				print(*req_dimension['default'], sep = 'x')
			exit(0)
		print('Check succesful')
		


	def build_model(self):
		raise NotImplementedError
