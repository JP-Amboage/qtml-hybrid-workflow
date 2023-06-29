from sklearn.svm import NuSVR
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Predictor():
	def __init__(self):
		self.trained = False
		self.model = NuSVR()
	
	def train(self,data, n_features):
		self.trained = True
		data = np.array(data)
		X = data[:,:n_features]
		y = data[:,-1]

		self.X_scaler = MinMaxScaler()
		X = self.X_scaler.fit_transform(X)
		self.y_scaler = MinMaxScaler()
		y = self.y_scaler.fit_transform(y.reshape(-1,1)).ravel()

		self.model.fit(X,y)

	def predict(self, l):
		scaled_input = self.X_scaler.transform(np.array(l).reshape(1, -1))
		scaled_prediction = self.model.predict(scaled_input).reshape(-1, 1)
		result = self.y_scaler.inverse_transform(scaled_prediction)[0][0]
		return result