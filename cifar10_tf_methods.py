'''
Mostly taken from RayTune docs: 
https://docs.ray.io/en/latest/tune/examples/includes/pbt_tune_cifar10_with_keras.html
'''
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import random

from scipy.stats import loguniform
from scipy.stats import uniform

#NUM_SAMPLES = 128 #use this for debuging (small training data) FAST
NUM_SAMPLES = None #use this in real situations (all trainng data) SLOW
num_classes = 10


class ConfigGeneratorCifar10():
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
				"learning_rate" : loguniform.rvs(1e-4, 1e-1),
				"dropout" : uniform.rvs(0,1),
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
	

def train_cifar(config: dict, id: str, epochs: int, dir_name: str):
	tf.keras.backend.clear_session()
	gpus = tf.config.list_physical_devices('GPU') 
	tf.config.set_visible_devices(gpus[0], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True) 
	
	model_file = "./"+dir_name+"/cifar_"+id+".h5"
	model = None

	if os.path.exists(model_file):
		model = load_model(model_file)
	else:
		model = build_model(config)
	
	train_data, test_data = read_data()
	x_train, y_train = train_data
	x_train, y_train = x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
	x_test, y_test = test_data
	x_test, y_test = x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]

	aug_gen = ImageDataGenerator(
		# set input mean to 0 over the dataset
		featurewise_center=False,
		# set each sample mean to 0
		samplewise_center=False,
		# divide inputs by dataset std
		featurewise_std_normalization=False,
		# divide each input by its std
		samplewise_std_normalization=False,
		# apply ZCA whitening
		zca_whitening=False,
		# randomly rotate images in the range (degrees, 0 to 180)
		rotation_range=0,
		# randomly shift images horizontally (fraction of total width)
		width_shift_range=0.1,
		# randomly shift images vertically (fraction of total height)
		height_shift_range=0.1,
		# randomly flip images
		horizontal_flip=True,
		# randomly flip images
		vertical_flip=False,
	)

	aug_gen.fit(x_train)
	batch_size = config.get("batch_size", 64)
	gen = aug_gen.flow(x_train, y_train, batch_size=batch_size)

	#losses = []
	#accs = []

	if epochs > 0:
		history = model.fit(gen, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
		# loss, accuracy
		#loss, acc= model.evaluate(x_test, y_test, verbose=0)
		#losses.append(loss)
		#accs.extend(list(-1*np.array(history.history['val_loss'])))
		model.save(model_file)
		return list(-1*np.array(history.history['val_accuracy']))
	del model
	return []

def read_data():
	# The data, split between train and test sets:
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# Convert class vectors to binary class matrices.
	y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	x_train = x_train.astype("float32")
	x_train /= 255
	x_test = x_test.astype("float32")
	x_test /= 255

	return (x_train, y_train), (x_test, y_test)

def build_model(config: dict):
		x = Input(shape=(32, 32, 3))
		y = x
		y = Convolution2D(
			filters=64,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = Convolution2D(
			filters=64,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)

		y = Convolution2D(
			filters=128,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = Convolution2D(
			filters=128,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)

		y = Convolution2D(
			filters=256,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = Convolution2D(
			filters=256,
			kernel_size=3,
			strides=1,
			padding="same",
			activation="relu",
			kernel_initializer="he_normal",
		)(y)
		y = MaxPooling2D(pool_size=2, strides=2, padding="same")(y)

		y = Flatten()(y)
		y = Dropout(config.get("dropout", 0.5))(y)
		y = Dense(units=10, activation="softmax", kernel_initializer="he_normal")(y)

		model = Model(inputs=x, outputs=y, name="model1")

		opt = tf.keras.optimizers.Adadelta(
			learning_rate=config.get("learning_rate", 1e-4), weight_decay=config.get("weight_decay", 1e-4)
		)
		model.compile(
			loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		return model
	