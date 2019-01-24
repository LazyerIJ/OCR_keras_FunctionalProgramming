# -*- coding: utf-8 -*-
'''This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.

This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.

The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, hence the different results.

                                                Norm. ED
Epoch |   TF   |   TH
------------------------
                10               0.027   0.064
                15               0.038   0.035
                20               0.043   0.045
                25               0.014   0.019

This requires cairo and editdistance packages:
pip install cairocffi
pip install editdistance

Created by Mike Henry
https://github.com/mbhenry/
'''
import json
import datetime
import matplotlib
import keras.backend as K
matplotlib.use('Agg')

from utils.helpers import *
from model import Model

K.set_image_data_format('channels_last')
params_utils = json.loads(open('utils/params.json').read())['utils']
np.random.seed(params_utils['seed'])

if __name__=='__main__':
    name = datetime.datetime.now().strftime(params_utils['name_format'])
    model = Model(name)
    model.fit()
