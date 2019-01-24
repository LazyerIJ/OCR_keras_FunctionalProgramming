import os
import json
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras import backend as K
from utils.helpers import params_train, params_data, params_utils
from utils.helpers import ctc_lambda_func

class modelCRNN():

    def __init__(self, input_shape, act='relu'):

        self.conv_filters = params_train['conv_filters']
        self.kernel_size = params_train['kernel_size']
        self.pool_size = params_train['pool_size']
        self.rnn_size = params_train['rnn_size']
        self.time_dense_size = params_train['time_dense_size']

        self.absolute_max_string_len = params_data['absolute_max_string_len'],
        self.output_size = len(params_utils['alphabet'])+1

        self.mono_file = params_data['monogram_file']
        self.bi_file = params_data['bigram_file']

        self.input_shape = input_shape
        self.act = act
        self.optimizer = SGD(lr=params_train['sgd']['lr'],
                             decay=params_train['sgd']['decay'],
                             momentum=params_train['sgd']['momentum'],
                             nesterov=params_train['sgd']['nesterov'],
                             clipnorm=params_train['sgd']['clipnorm'])

        self.init_model()


    def init_model(self):
        self.input_shape[1]

        input_data = Input(name='the_input', 
                           shape=self.input_shape, 
                           dtype='float32')
        inner = Conv2D(self.conv_filters,
                       self.kernel_size,
                       padding='same',
                       activation=self.act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size),
                             name='max1')(inner)

        inner = Conv2D(self.conv_filters,
                       self.kernel_size,
                       padding='same',
                       activation=self.act,
                       kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size),
                             name='max2')(inner)

        inner = Conv2D(self.conv_filters,
                       self.kernel_size,
                       padding='same',
                       activation=self.act,
                       kernel_initializer='he_normal',
                       name='conv3')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size),
                             name='max3')(inner)

        conv_to_rnn_dims = (self.input_shape[0] // (self.pool_size ** 3),
                            (self.input_shape[1] // (self.pool_size ** 3)) * self.conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.time_dense_size, activation=self.act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.rnn_size, return_sequences=True, go_backwards=True, 
                     kernel_initializer='he_normal', name='gru1_b')(inner)
        '''
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.rnn_size, return_sequences=True,
                    kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
        '''
        # transforms RNN output to character activations:
        inner = Dense(self.output_size, kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_1, gru_1b]))
        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels',
                       shape=[self.absolute_max_string_len[0]], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(
                    self.ctc_lambda_func, output_shape=(1,),
                    name='ctc')([y_pred, labels, input_length, label_length])

        self.crnn = Model(inputs=[input_data, labels, input_length, label_length],
                           outputs=loss_out)
        self.test_func = K.function([input_data], [y_pred])
        self.crnn.compile(loss={'ctc' : lambda y_true, y_pred : y_pred},
                           optimizer=self.optimizer)

    def ctc_lambda_func(self, args):
            y_pred, labels, input_length, label_length = args
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage:
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

