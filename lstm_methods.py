'''
Mostly taken from RayTune docs: 
https://docs.ray.io/en/latest/tune/examples/includes/pbt_memnn_example.html
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from scipy.stats import loguniform
from scipy.stats import uniform
import tarfile
import re
import numpy as np
import os

class ConfigGeneratorLSTM():
	def __init__(self, random_state=None):
		if random_state == None:
			random_state = random.randint(0,9999)
		random.seed(random_state)
		np.random.seed(random_state)
		#tf.random.set_seed(random_state)
		#tf.keras.utils.set_random_seed(random_state) 
		#os.environ['PYTHONHASHSEED'] = str(random_state)
		#os.environ['TF_DETERMINISTIC_OPS'] = '1'
		
		self.n_sampled = 0
	
	def get_hyperparameter_configuration(self, n):
		'''
		returns n configurations
		'''
		T = []
		for _ in range(n):
			config = {
				"lr" : 10 ** (-1*uniform.rvs(0,10)),
				"dropout" : uniform.rvs(0,1),
				"rho": uniform.rvs(0,1),
				"weight_decay" : loguniform.rvs(1e-5,0.1)
			}
			id = str(self.n_sampled)
			t = {
				"config": config,
				"id": id,
				"curve": []
			}
			T.append(t)
			self.n_sampled = self.n_sampled + 1
		return T

def train_LSTM(config: dict, id: str, epochs: int, dir_name: str):
	tf.keras.backend.clear_session()
	gpus = tf.config.list_physical_devices('GPU') 
	tf.config.set_visible_devices(gpus[0], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True) 


	inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, vocab_size, story_maxlen, query_maxlen = read_data()
	
	model_file = "./"+dir_name+"/lstm_"+id+".h5"
	model = None



	if os.path.exists(model_file):
		model = load_model(model_file)
	else:
		model = build_model(config, vocab_size=vocab_size, story_maxlen=story_maxlen, query_maxlen=query_maxlen)

	#losses = []
	accs = []

	if epochs > 0:
		for i in range(epochs):
			model.fit(
				[inputs_train, queries_train],
				answers_train,
				batch_size=config.get("batch_size", 32),
				epochs=1,
				validation_data=([inputs_test, queries_test], answers_test),
				verbose=0,
			)

			_, accuracy = model.evaluate(
			[inputs_test, queries_test], answers_test, verbose=0
			)
			accs.append(-1.0*accuracy)
		model.save(model_file)
		return accs[:]
	del model
	return []

###########################################################################################################
######################################### AUXILIAR METHODS ################################################
###########################################################################################################
def build_model(config, vocab_size, story_maxlen, query_maxlen ):
	"""Helper method for creating the model"""

	# placeholders
	input_sequence = Input((story_maxlen,))
	question = Input((query_maxlen,))

	# encoders
	# embed the input sequence into a sequence of vectors
	input_encoder_m = Sequential()
	input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
	input_encoder_m.add(Dropout(config.get("dropout", 0.3)))
		# output: (samples, story_maxlen, embedding_dim)

	# embed the input into a sequence of vectors of size query_maxlen
	input_encoder_c = Sequential()
	input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
	input_encoder_c.add(Dropout(config.get("dropout", 0.3)))
	 # output: (samples, story_maxlen, query_maxlen)

	# embed the question into a sequence of vectors
	question_encoder = Sequential()
	question_encoder.add(
		Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen)
	)
	 
	question_encoder.add(Dropout(config.get("dropout", 0.3)))
	# output: (samples, query_maxlen, embedding_dim)

	# encode input sequence and questions (which are indices)
	# to sequences of dense vectors
	input_encoded_m = input_encoder_m(input_sequence)
	input_encoded_c = input_encoder_c(input_sequence)
	question_encoded = question_encoder(question)

	# compute a "match" between the first input vector sequence
	# and the question vector sequence
	# shape: `(samples, story_maxlen, query_maxlen)`
	match = dot([input_encoded_m, question_encoded], axes=(2, 2))
	match = Activation("softmax")(match)

	# add the match matrix with the second input vector sequence
	response = add(
		[match, input_encoded_c]
	)  # (samples, story_maxlen, query_maxlen)
	response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

	# concatenate the match matrix with the question vector sequence
	answer = concatenate([response, question_encoded])

		# the original paper uses a matrix multiplication.
		# we choose to use a RNN instead.
	answer = LSTM(32)(answer)  # (samples, 32)

	# one regularization layer -- more would probably be needed.
	answer = Dropout(config.get("dropout", 0.3))(answer)
	answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
	# we output a probability distribution over the vocabulary
	answer = Activation("softmax")(answer)

	# build the final model
	model = Model([input_sequence, question], answer)

	rmsprop = RMSprop(
			lr=config.get("lr", 1e-3), rho=config.get("rho", 0.9))
	
	model.compile(
			optimizer=rmsprop,
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"],
	)
	
	return model
	

def tokenize(sent):
	"""Return the tokens of a sentence including punctuation.

	>>> tokenize("Bob dropped the apple. Where is the apple?")
	["Bob", "dropped", "the", "apple", ".", "Where", "is", "the", "apple", "?"]
	"""
	return [x.strip() for x in re.split(r"(\W+)?", sent) if x and x.strip()]

def parse_stories(lines, only_supporting=False):
	"""Parse stories provided in the bAbi tasks format

	If only_supporting is true, only the sentences
	that support the answer are kept.
	"""
	data = []
	story = []
	for line in lines:
		line = line.decode("utf-8").strip()
		nid, line = line.split(" ", 1)
		nid = int(nid)
		if nid == 1:
			story = []
		if "\t" in line:
			q, a, supporting = line.split("\t")
			q = tokenize(q)
			if only_supporting:
				# Only select the related substory
				supporting = map(int, supporting.split())
				substory = [story[i - 1] for i in supporting]
			else:
				# Provide all the substories
				substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append("")
		else:
			sent = tokenize(line)
			story.append(sent)
	return data

def get_stories(f, only_supporting=False, max_length=None):
	"""Given a file name, read the file,
	retrieve the stories,
	and then convert the sentences into a single story.

	If max_length is supplied,
	any stories longer than max_length tokens will be discarded.
	"""

	def flatten(data):
		return sum(data, [])

	data = parse_stories(f.readlines(), only_supporting=only_supporting)
	data = [
		(flatten(story), q, answer)
		for story, q, answer in data
		if not max_length or len(flatten(story)) < max_length
	]
	return data

def vectorize_stories(word_idx, story_maxlen, query_maxlen, data):
	inputs, queries, answers = [], [], []
	for story, query, answer in data:
		inputs.append([word_idx[w] for w in story])
		queries.append([word_idx[w] for w in query])
		answers.append(word_idx[answer])
	return (
		pad_sequences(inputs, maxlen=story_maxlen),
		pad_sequences(queries, maxlen=query_maxlen),
		np.array(answers),
	)
def read_data(challenge_type = "QA17"):
	# Get the file
	try:
		#/p/home/jusers/garciaamboage1/deep/.keras/datasets/babi-tasks-v1-2.tar.gz
		path = get_file(
			"babi-tasks-v1-2.tar.gz",
			origin="https://s3.amazonaws.com/text-datasets/"
			"babi_tasks_1-20_v1-2.tar.gz",
		)
	except Exception:
		print(
			"Error downloading dataset, please download it manually:\n"
			"$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2"  # noqa: E501
			".tar.gz\n"
			"$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz"  # noqa: E501
		)
		raise

	# Choose challenge
	challenges = {
		# QA1 with 10,000 samples
		"QA1": "tasks_1-20_v1-2/en-10k/qa1_"
		"single-supporting-fact_{}.txt",
		# QA2 with 10,000 samples
		"QA2": "tasks_1-20_v1-2/en-10k/qa2_"
		"two-supporting-facts_{}.txt",

		"QA3": "tasks_1-20_v1-2/en-10k/qa3_"
		"three-supporting-facts_{}.txt",

		"QA4": "tasks_1-20_v1-2/en-10k/qa4_"
		"two-arg-relations_{}.txt",

		"QA5": "tasks_1-20_v1-2/en-10k/qa5_"
		"three-arg-relations_{}.txt",

		"QA6": "tasks_1-20_v1-2/en-10k/qa6_"
		"yes-no-questions_{}.txt",

		"QA7": "tasks_1-20_v1-2/en-10k/qa7_"
		"counting_{}.txt",

		"QA8": "tasks_1-20_v1-2/en-10k/qa8_"
		"lists-sets_{}.txt",

		"QA9": "tasks_1-20_v1-2/en-10k/qa9_"
		"simple-negation_{}.txt",

		"QA10": "tasks_1-20_v1-2/en-10k/qa10_"
		"indefinite-knowledge_{}.txt",

		"QA11": "tasks_1-20_v1-2/en-10k/qa11_"
		"basic-coreference_{}.txt",

		"QA12": "tasks_1-20_v1-2/en-10k/qa12_"
		"conjunction_{}.txt",

		"QA13": "tasks_1-20_v1-2/en-10k/qa13_"
		"compound-coreference_{}.txt",

		"QA14": "tasks_1-20_v1-2/en-10k/qa14_"
		"time-reasoning_{}.txt",

		"QA15": "tasks_1-20_v1-2/en-10k/qa15_"
		"basic-deduction_{}.txt",

		"QA16": "tasks_1-20_v1-2/en-10k/qa16_"
		"basic-induction_{}.txt",

		"QA17": "tasks_1-20_v1-2/en-10k/qa17_"
		"positional-reasoning_{}.txt",

		"QA18": "tasks_1-20_v1-2/en-10k/qa18_"
		"size-reasoning_{}.txt",

		"QA19": "tasks_1-20_v1-2/en-10k/qa19_"
		"path-finding_{}.txt",

		"QA20": "tasks_1-20_v1-2/en-10k/qa20_"
		"agents-motivations_{}.txt",
	}
	if challenge_type == None: challenge_type = "QA1"
	challenge = challenges[challenge_type]

	with tarfile.open(path) as tar:
		train_stories = get_stories(tar.extractfile(challenge.format("train")))
		test_stories = get_stories(tar.extractfile(challenge.format("test")))

	vocab = set()
	for story, q, answer in train_stories + test_stories:
		vocab |= set(story + q + [answer])
	vocab = sorted(vocab)

	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	story_maxlen = max(len(x) for x, _, _ in train_stories + test_stories)
	query_maxlen = max(len(x) for _, x, _ in train_stories + test_stories)

	word_idx = {c: i + 1 for i, c in enumerate(vocab)}
	inputs_train, queries_train, answers_train = vectorize_stories(
		word_idx, story_maxlen, query_maxlen, train_stories
	)
	inputs_test, queries_test, answers_test = vectorize_stories(
		word_idx, story_maxlen, query_maxlen, test_stories
	)

	return inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test, vocab_size, story_maxlen, query_maxlen
