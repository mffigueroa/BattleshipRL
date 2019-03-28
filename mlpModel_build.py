import keras
from keras import regularizers
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Lambda

def CombineValueAndAdvantage(input):
	value = input[:,0]
	advantage = input[:,1:]
	mean_advantage = K.mean(advantage, axis=-1)
	reshapedAdvantageOffset = K.reshape(value - mean_advantage, (K.shape(advantage)[0],1))
	return advantage + reshapedAdvantageOffset
	
def BuildMLPModel(maxSeqLength, inputDimension, outputDimension):
	model = Sequential()
	inputDimensions = int(maxSeqLength*inputDimension)
	model.add(Dense(inputDimensions // 2, activation='tanh', input_shape=(inputDimensions,), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dense(inputDimensions // 4, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dense(inputDimensions // 2, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dense(outputDimension + 1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Lambda(CombineValueAndAdvantage))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','acc'])
	return model