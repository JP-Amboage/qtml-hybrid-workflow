import json
from hashlib import md5
import itertools
import numpy as np

def fun_worker(rank, dir_name, comm, train_model_method):
	# function for the worker nodes
	while True:
		# receive a configuration to train
		message = comm.recv(None, source=0)
		if message == {}: break
		# receive config
		trial = message['trial']
		#print("Worker {} received config {}".format(rank, trial['id']))
		config = trial['config']
		initial = message['iteration'] == 0
		# run training
		val_acc = train_model_method(config=config, id=trial['id'], epochs = message['epochs'], dir_name = dir_name)
		iteration = message['iteration'] + 1 
			
		# concat learning curve if necessary
		if trial['curve'] is not None:
			trial['curve'].extend(val_acc)
		else:
			trial['curve'] = val_acc
			
		# send back results to head node
		comm.send({'trial': trial, 'iteration': iteration}, dest=0, tag=rank)
		#print("Worker {} done with config {}".format(rank, trial["id"]))

def send_stop_all_workers(size: int, comm) -> None:
	for worker in range(1,size):
		comm.send({}, dest=worker)

def dict_hash(dictionary) -> str:
	"""MD5 hash of a dictionary."""
	dhash = md5()
	#dhash.update(str(frozenset(dictionary.values())).encode())
	dhash.update(json.dumps(dictionary, sort_keys=True).encode())
	return dhash.hexdigest()

def top_k(configs, losses, k):
	'''
	takes a set of configurations as well as their associated losses and returns the top k performing configurations
	'''
	return [configs[idx] for idx in np.argsort(losses)[:k]]

def config_with_smallest_loss(T):
	best_loss = np.inf
	best_config = []
	for config in  itertools.chain(*T):
		if config['curve'][-1] <  best_loss:
			best_loss = config['curve'][-1]
			best_config=[config]
		elif config['curve'][-1] == best_loss:
			best_config.append(config)
	return best_config