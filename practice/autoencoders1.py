import keras
from keras.layers import Input, Dense
from keras.models import Model
import os
import numpy as np 
import pandas as pd 


class Autoencoder :

	def __init__(self,dataset = None, comp_factor = 2) :
		self.model = None # autoencoder neural network
		self.X = dataset
		self.comp_factor = comp_factor # compression factor - Factor by which the no of nodes reduce after each layer

		self.input_layer_units = self.X.shape[1] # No of nodes in the input layer
		self.compressed_layer_units = 0 # No of layers in the compresses layer- Latent Representation

	def _encoder(self) : #creates the encoder model

		input = Input(shape = (self.X.shape[1],))

		layer1_units = round(self.X.shape[1]/2)

		layer1 = Dense(layer1_units, activation = 'relu')(input)

		layer2_units = round(layer1_units/2)
		self.compressed_layer_units = layer2_units

		layer2 = Dense(layer2_units, activation = 'relu')(layer1)

		compressed_layer = layer2

		model = Model(input, compressed_layer)

		return model

	def _decoder(self) : # it takes the compressed layer as the input layer and reconstructs the dataset from this compresses layer
					 #its output layer is the same number of nodes as the input layer of the encoder
		input = Input(shape = (self.compressed_layer_units, ))

		layer1_units = self.compressed_layer_units*2
		layer1 = Dense(layer1_units, activation = 'relu')(input)

		layer2_units = self.input_layer_units
		layer2 = Dense(layer2_units, activation = 'relu')(layer1)

		reconstructed_layer = layer2

		model = Model(input, reconstructed_layer)

		return model


	def build_model(self) : # this function combines the encoder and decoder

		self.encoder = self._encoder()
		self.decoder = self._decoder()

		input = Input(shape = (self.input_layer_units, ))

		enc_out  = self.encoder(input)

		dec_out = self.decoder(enc_out)

		model = Model(input, dec_out)

		self.model = model

		return model


	def fit(self, batch_size=20, epochs=300) : #compiles and fits the model

		self.model.compile(optimizer = 'sgd', loss = 'mse', metrics = ['accuracy'])

		self.model.fit(self.X, self.X, batch_size = batch_size, epochs = epochs)


	def save(self) :

		if not os.path.exists(r'./weights1') :
			os.mkdir(r'./weights1')

		self.encoder.save(r'./weights1/enc_weights.h5')
		self.decoder.save(r'./weights1/dec_weights.h5')
		self.model.save(r'./weights1/autoencoder_weights.h5')



def main() :

	#data preprocessing
	df = pd.read_csv('/home/safeer/Documents/QCRI/datasets/Absenteeism_at_work.csv', sep = ';')
	df = df.iloc[:,5:]
	X = df.iloc[:,:].values

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X = sc.fit_transform(X)


	autoEnc_model = Autoencoder(dataset = X)
	autoEnc_model.build_model()
	autoEnc_model.fit()




if __name__ == '__main__':
	main()




		