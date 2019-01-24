"""
Make model with crnn + generator
"""
import os
from models.CRNN import *
from datas.Generator import *
from utils.vis_callback import *
from utils.helpers import params_utils, params_data, params_train

class Model():

    def __init__(self, name):

        self.name = name
        self.output_dir = params_utils['output_dir']
        self.save_format = params_utils['save_file_format']
        self.minibatch_size = params_train['minibatch_size']
        self.words_per_epoch = params_train['words_per_epoch']
        self.val_split = params_train['val_split']
        self.val_words = int(self.words_per_epoch * (self.val_split))
        self.train_step = params_utils['train_step']
        self.img_w=params_data['img_w']

        steps_per_epoch = (self.words_per_epoch - self.val_words)
        self.steps_per_epoch = steps_per_epoch // self.minibatch_size

    def init_gen(self, input_shape):
        self.img_gen = TextImageGenerator(img_w=input_shape[0],
                                          img_h=input_shape[1],
                                          val_split = self.words_per_epoch-self.val_words)
    def init_model(self, input_shape):
        self.model = modelCRNN(input_shape)
        self.viz_cb = VizCallback(self.name,
                                  self.model.test_func,
                                  self.img_gen.next_val())
        print('\n')
        print(self.model.crnn.summary())
        print('\n')

    def load_weights(self, weight_file):
        file = os.path.join(self.output_dir,
                            os.path.join(self.name,self.save_format%(start - 1)))
        self.model.load_weights(file)

    def fit(self):

        for idx in range(1, len(self.train_step)):
            start_epoch = self.train_step[idx-1]
            stop_epoch = self.train_step[idx]
            load_weight = False if start_epoch==0 else True
            input_shape = (self.img_w[idx-1], 64, 1)

            self.init_gen(input_shape)
            self.init_model(input_shape)

            if load_weight:
                self.load_weights()


            self.model.crnn.fit_generator(
                generator=self.img_gen.next_train(),
                steps_per_epoch=self.steps_per_epoch,
                epochs=stop_epoch,
                initial_epoch=start_epoch,
                validation_data=self.img_gen.next_val(),
                validation_steps=self.val_words // self.minibatch_size,
                callbacks=[self.viz_cb, self.img_gen],
            )
