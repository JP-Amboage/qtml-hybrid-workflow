import numpy as np
from sklearn.preprocessing import MinMaxScaler
from QSVR_extended import QSVR
import random
import os
import utility as utility
import embedding as embedding

class QPredictor():
	def __init__(self, sampler = None):
		os.environ["DWAVE_API_TOKEN"] = "xNrg-2f975ece26fc5e95dc26d687de6668a96ea0d4c6"
		self.trained = False
		self.model = QSVR()
		self.sampler = sampler 
		
	def train(self,data, n_features):
		
		self.trained = True
		data = np.array(data)
		X = data[:,:n_features]
		y = data[:,-1]
		
		self.X_scaler = MinMaxScaler()
		X = self.X_scaler.fit_transform(X)
		self.y_scaler = MinMaxScaler()
		y = self.y_scaler.fit_transform(y.reshape(-1,1))

		
		if X.shape[0] > 20:
			idxs = random.sample(range(0,X.shape[0]),20)
			#print("######################################################Reduced training points")
			X = X[idxs,]
			y = y[idxs,]

		K = 3

		problem_size = X.shape[0] * 2 * K

		if problem_size > 120:
			exit(0)

		if self.sampler == None:
			if not os.path.isfile("./embeddings/" + str(problem_size) + "embedding.txt"):
				print("=== Couldn't find a suitable embedding file: A new embedding file will be created: " + str(problem_size) + "embedding.txt")
				emb = embedding.get_clique_emebedding(dim=problem_size,region="eu-central-1", solver="Advantage_system5.4")
				if not os.path.exists("./embeddings"): os.makedirs("./embeddings")
				embedding.save_embedding(emb, "./embeddings/" + str(problem_size))
		
			embedding_file_name = "./embeddings/" + str(problem_size) + "embedding.txt"
			region='eu-central-1'
			solver='Advantage_system5.4'
			self.sampler = utility.define_sampler_from_embedding(embedding_file_name, region, solver)

		self.model.fit(X, y.reshape(-1,1),
			K = K, B = 0.5,
			epsilon = 0.02, k0 = 0.005,
			xi=0.01, n_samples = X.shape[0], num_reads = 1000,
			random_seed=0,
			n_samples_for_gamma_and_C_optimizations=0,
			gamma=0.1, C=67.61,
			use_custom_chainstrength=True,
			chain_mult=10,
			sampler=self.sampler
		)
		
		self.X = X
		self.y = y

	def predict(self, l):
		return self.y_scaler.inverse_transform(np.sum(self.model.predict(self.X_scaler.transform(np.array(l).reshape(1, -1))), axis=0).reshape(-1, 1)/7)[0][0]
	

if __name__ == "__main__":
		import pandas as pd
		from timeit import default_timer as timer
		print("imported pandas")
		df = pd.read_csv("mlpf.csv")
		print("read dataset")
		data = [ [df.iloc[i]['loss_'+str(j)] for j in range(40)] for i in range(30) ]
		print("prepared training data")
		predictor = QPredictor()
		print("created predictor")
		start = timer()
		predictor.train(data=data, n_features=15)
		end = timer()
		print(f"trained predictor in {end-start} seconds")
		print(predictor.predict([df.iloc[50]['loss_'+str(j)] for j in range(15)]))
		print("finished")
	
