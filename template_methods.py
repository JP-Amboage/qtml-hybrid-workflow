class ConfigGeneratorTemplate():
	def __init__(self, random_state=None):
		self.n_sampled = 0

	def get_hyperparameter_configuration(self, n):
		'''
		returns n configurations
		'''
		T = []
		for _ in range(n):
			config = {} # config dict
			id = 0 # unique config id
			t = {
				"config": config,
				"id": id,
				"curve": []
			}
			T.append(t)
			self.n_sampled = self.n_sampled + 1
		return T

def trainTemplate(config: dict, id: str, epochs: int, dir_name: str) -> list:
	'''
	If a model with the given id already exists in the given dir the model is loaded
	Otherwise a model with the given config is created
	The model is trained for the given number of epochs
	The model is saved in the given dir
	A list with len=epochs containing the loss ( or -1 * accuracy ) of the model
		after each of the traning epochs is returned
	'''
	return []