from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, CuDNNLSTM, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam
    
def normalized_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))
    
def build_model(vocab_size, input_size, embedding_dim, lstm_units, embedding_matrix, learning_rate = 0.001, clip_norm = None):
    left_input = Input(shape=(input_size,), dtype='int32')
    
    right_input = Input(shape=(input_size,), dtype='int32')
    
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_size, trainable=False)
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    lstm = CuDNNLSTM(lstm_units, unit_forget_bias = 1, kernel_initializer = TruncatedNormal())

    left_hidden = lstm(encoded_left)
    right_hidden = lstm(encoded_right)

    similarity = Lambda(function=lambda x: normalized_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_hidden, right_hidden])

    model = Model([left_input, right_input], [similarity])
    
    if clip_norm:
        optimizer = Adam(lr = learning_rate, clipnorm = clip_norm)
    else:
        optimizer = Adam(lr = learning_rate)
        
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model