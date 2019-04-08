import keras
from keras import regularizers
import keras.backend as K
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Dense, Lambda

def CombineValueAndAdvantage(input):
	value = input[:,0]
	advantage = input[:,1:]
	mean_advantage = K.mean(advantage, axis=-1)
	reshapedAdvantageOffset = K.reshape(value - mean_advantage, (K.shape(advantage)[0],1))
	return advantage + reshapedAdvantageOffset
	
def BuildMLPModel(maxSeqLength, inputDimension, outputDimension):
	inputDimensions = int(maxSeqLength*inputDimension)
	inputLayer = Input(shape=(inputDimensions,))
	denseLayer = Dense(inputDimensions // 2, activation='tanh', input_shape=(inputDimensions,), kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputLayer)
	denseLayer = Dense(inputDimensions // 4, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros')(denseLayer)
	denseLayer = Dense(inputDimensions // 2, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros')(denseLayer)
	denseLayer = Dense(outputDimension + 1, activation='linear', kernel_initializer='glorot_uniform', bias_initializer='zeros')(denseLayer)
	valueAdvantage = Lambda(CombineValueAndAdvantage)(denseLayer)
	
	model = Model(inputs=inputLayer, outputs=valueAdvantage)
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','acc'])
	return model