import shutil
from mpi4py import MPI
import platform
import click
from timeit import default_timer as timer

from math import floor
from math import ceil
from math import log

import numpy as np

from utils_MPI import fun_worker
from utils_MPI import send_stop_all_workers
from utils_MPI import top_k
from utils_MPI import config_with_smallest_loss

EPOCHS = 0
Predictor_class = None

def run_then_return_validation_loss(comm, T, size, r_prev, r_i):
	global EPOCHS 
	avail_workers = [i for i in range(1,size)]
	all_workers = [i for i in range(1,size)]
	index = 0
	FULLY_TRAINED = []
	n_fully_trained = 0
	n_sent_to_fully_train = 0
	# FULLY TRAIN A MINIMUM of TRIALS and PARTIALLY TRAIN the rest
	while(True):
		# if more workers available than experiments left, remove workers
		while (index + len(avail_workers) > len(T)):
			removed_worker = avail_workers.pop()
			all_workers.remove(removed_worker)
		
		# send a partial train to every available worker
		for worker in avail_workers:
			#print("Inital sending config id {} to worker {}".format(T[index]["id"], worker))
			comm.send({"trial": T[index], "iteration": 0, "epochs": r_i-r_prev}, dest=worker)
			EPOCHS = EPOCHS + r_i-r_prev
			index = index + 1
		# afterwards no worker is available
		avail_workers = []
	
		# we will receive an answer for every working worker
		for _ in all_workers:
			status = MPI.Status()
			# collect the answer
			message = comm.recv(None, source=MPI.ANY_SOURCE, status=status)
			# collect the sender
			source = status.Get_source()
			#print("Received val acc {} from rank {}".format(message["trial"]['curve'][-1], source))

			#locate  and update the trial
			for i in range(len(T)): 
				if T[i]['id'] == message['trial']['id']:
					T[i] = message['trial']
					t = T[i]
					break

			# if the answer if the result of a fully trained
			n_fully_trained = n_fully_trained + 1
			#print("Trial with id {} finished with acc {}".format(t['id'], t['curve'][-1]))
			# save config+learning curve in the table that will be used for training the SVR    
			lr_curve =  t['curve']
			# save full config + curve in list
			# the worker is released
			avail_workers.append(source)
			FULLY_TRAINED.append(t)
			
		if (index >= len(T) and set(avail_workers) == set(all_workers)):
			#print("Stop")
			#print(index)
			break
	
	# losses list in the order of the Trials
	L = [np.inf]*len(T)

	for i in range(len(T)):
		#if the trial only was partially trained we set a high loss so that it gets killed
		if len(T[i]['curve']) >= r_i:
			L[i] = T[i]['curve'][-1]
		else:
			print("!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!")
	return L, T
			
def hyperband(
		R, # Max resources allocated to any configuration 
		eta, # controls the proportion of configurations discarded in each round of SuccessiveHalving
		configGenerator,
		comm, 
		size
	):
		s_max = floor(log(R,eta))
		B = (s_max + 1) * R

		selected = []
		for s in range(s_max, -1, -1):
			n = ceil((B/R)*((eta**s)/(s+1)))
			r = R*(eta**-s)
			# Begin SuccessiveHalving with (n,r) inner loop
			print(f"=====Begin SuccessiveHalving with {(n,r)} inner loop=====")
			T = configGenerator.get_hyperparameter_configuration(n)
			r_prev = 0
			for i in range(0, s+1, 1):
				n_i = floor(n*(eta**-i))
				r_i = round(r *(eta**i))
				print((n_i,r_i))
				L, T =  run_then_return_validation_loss(comm=comm, T=T, size=size, r_prev=r_prev, r_i=r_i)
				if len(L)!=len(T): print("PROBLEM!!!!")
				#print(T)
				T = top_k(T, L, max(floor(n_i/eta),1))
				r_prev = r_i
			selected.append(T)
		
		selected = selected
		return config_with_smallest_loss(selected)



@click.command()
@click.option('--dir_name', default="saved_models_regular", help='name of folder where models are saved')
@click.option('--seed', default=0, help="seed")
@click.option('--model_name', default='lstm', help="lstm | cifar10_tf :target model for the HPO process")
@click.option('--save_models', default=False, help="False | True")
@click.option('--r', default=100, help="R value for Hyperband")
@click.option('--eta', default=2, help="eta value for Hyperband")
def main(dir_name, seed, model_name, save_models,
		r, eta
	):
	global Predictor_class
	R=r
	
	'''
	CHANGE THIS TO ADD MORE MODELS
	'''
	if model_name == 'cifar10_tf':
		from cifar10_tf_methods import ConfigGeneratorCifar10
		ConfigGenerator_class = ConfigGeneratorCifar10
		from cifar10_tf_methods import train_cifar
		train_model_method = train_cifar
	else: 
		if model_name != 'lstm':
			print("WARNING: model_name not valid -> using lstm as default choice")
		from lstm_methods import ConfigGeneratorLSTM
		ConfigGenerator_class = ConfigGeneratorLSTM
		from lstm_methods import train_LSTM 
		train_model_method = train_LSTM
	'''
	END OF THE MODEL DEPENDENT PART
	'''

	comm = MPI.COMM_WORLD
	size = MPI.COMM_WORLD.Get_size()
	rank = MPI.COMM_WORLD.Get_rank()
	print(f'------> task {rank} running from {platform.node()}')
	if (rank == 0):
		rs = seed
		print(f"R={R}, eta={eta}, size={size}, rs={rs}")
		global EPOCHS
		EPOCHS = 0
		print("MPI WORLD SIZE: ", size)
		generator = ConfigGenerator_class(random_state=rs)	
		
		start = timer()
		res = hyperband(
				R=R, # Max resources allocated to any configuration 
				eta=eta, # controls the proportion of configurations discarded in each round of SuccessiveHalving
				configGenerator=generator,
				comm=comm,
				size=size
		)
		end = timer()

		print("#################################################################")
		print("#################################################################")
		# print out results
		print(f"Best loss found: {res[0]['curve'][-1]}")
		print(f"Needed time: {end-start}s")
		print(f"Total Epochs: {EPOCHS}")
		print(f"Config ID: {res[0]['id']}")
		print(f"R={R}, eta={eta}, size={size}, rs={rs}")
		print("#################################################################")
		print("#################################################################")
		
		if save_models != True:
			shutil.rmtree(dir_name)
		send_stop_all_workers(size, comm)
		
	else :
		fun_worker(rank=rank, dir_name=dir_name, comm=comm, train_model_method=train_model_method)

if __name__ == "__main__":
	main()
