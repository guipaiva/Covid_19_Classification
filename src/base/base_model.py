from utils.definitions import MODELS_SHAPE

class BaseModel(object):
	def __init__(self, im_shape, name, classes, transfer_learn):
		self.model = None
		self.im_shape = im_shape
		self.name = name
		self.classes = classes
		self.transfer_learn = transfer_learn

		try:
			req_dimension = MODELS_SHAPE[self.name]
			if self.transfer_learn:
				assert len(im_shape) == 3 and (im_shape >= req_dimension['min'])
			else:
				assert im_shape == req_dimension['default']
		except AssertionError:
			if self.transfer_learn:
				print('Error: Image required to have 3 channels and with and height should not be smaller than', req_dimension['min'][0])
			else:
				print('Error: Image shape without transfer learning must be', end = ' ')
				print(*req_dimension['default'], sep = 'x')
			exit(0)
	
	def build_model(self):
		raise NotImplementedError
