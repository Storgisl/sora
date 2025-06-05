from __future__ import print_function
import numpy as np
import os
import SharedArray as sa
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.datasets import mnist

class InputData:
    def __init__(self, model, batch_size=64, seed=None):
        self.model = model # to get endpoint
        self.batch_size = batch_size
        self.z = dict()
        self.x = dict()
        self.seed = seed
        np.random.seed(seed)
    def add_data(self, path_new, key='train'):
        self.x[key] = np.load(path_new)
        print('data size:', self.x[key].shape)

    def add_data_sa(self, path_new, key='train'):
        self.x[key] = sa.attach(path_new)
        print('data size:', self.x[key].shape)

    def add_data_np(self, data, key='train'):
        self.x[key] = data
        print('data size:', self.x[key].shape)

    def get_batch_num(self, key='train'):
        return len(self.x[key]) // self.batch_size

    def get_batch(self, idx=0, data_size=None, key='train'):
        data_size = self.batch_size if data_size is None else data_size
        st = self.batch_size*idx
        x = self.x[key][st:st+data_size] * 2. - 1.
        x = x.reshape((-1, 4, 4, 96, 84, 5))  # Adjust according to your actual intended dimensions
        return x


    def get_rand_smaples(self, sample_size=64, key='train'):
        random_idx = np.random.choice(len(self.x[key]), sample_size, replace=False)
        return self.x[key][random_idx]*2. - 1.

    def gen_feed_dict(self, idx=0, data_size=None, key='train', z=None):
        feed_dict = self.gen_z_dict(data_size=data_size, z=z)

        if key is not None:
            x = self.get_batch(idx, data_size, key)
            feed_dict[self.model.x] = x

        return feed_dict

#######################################################################################################################
# Image
#######################################################################################################################

class InputDataMNIST(InputData):
    dataset_dir = 'dataset/mnist/original'
    def __init__(self, model, batch_size=64):
        self.model = model # to get endpoint
        self.batch_size = batch_size
        self.x = dict()

        (train_images, _), (test_images, _) = mnist.load_data()
        train_images = train_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        test_images = test_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

        self.add_data_np(train_images, 'train')
        self.add_data_np(test_images, 'test')

    def gen_feed_dict(self, idx=0, data_size=None, key='train'):
        batch_size = self.batch_size if data_size is None else data_size
        z = np.random.uniform(-1., 1., size=(self.batch_size, self.model.z_dim)).astype(np.float32)
        x = self.get_batch(idx, data_size, key)

        feed_dict = {self.model.z: z, self.model.x: x}

        return feed_dict

#######################################################################################################################
# Music
#######################################################################################################################

# Nowbar
class InputDataNowBarHybrid(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['inter']=  np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
            self.z['intra'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
        z_dict = {self.model.z_intra: self.z['intra'], self.model.z_inter:self.z['inter']}
        return z_dict

class InputDataNowBarJamming(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['intra'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
        z_dict = {self.model.z_intra: self.z['intra']}
        return z_dict

class InputDataNowBarComposer(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['inter'] = np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
        z_dict = {self.model.z_inter: self.z['inter']}

        return z_dict

# temporal
class InputDataTemporalHybrid(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['z_intra_v'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
            self.z['z_intra_i'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
            self.z['z_inter_v'] = np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
            self.z['z_inter_i'] = np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)

        feed_dict = {self.model.z_intra_v:  self.z['z_intra_v'], self.model.z_intra_i: self.z['z_intra_i'],
                    self.model.z_inter_v: self.z['z_inter_v'], self.model.z_inter_i: self.z['z_inter_i']}

        return feed_dict

class InputDataTemporalJamming(InputData):
    def gen_z_dict(self, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['z_intra_v'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)
            self.z['z_intra_i'] = np.random.normal(0, 0.1, [batch_size, self.model.z_intra_dim, self.model.track_dim]).astype(np.float32)

        feed_dict = {self.model.z_intra_v: self.z['z_intra_v'], self.model.z_intra_i: self.z['z_intra_i']}

        return feed_dict

class InputDataTemporalComposer(InputData):
    def gen_z_dict(self, idx=0, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['z_inter_v'] = np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
            self.z['z_inter_i'] = np.random.normal(0, 0.1, [batch_size, self.model.z_inter_dim]).astype(np.float32)
        feed_dict = {self.model.z_inter_v: self.z['z_inter_v'], self.model.z_inter_i: self.z['z_inter_i']}

        return feed_dict

class InputDataRNNComposer(InputData):
    def gen_feed_dict(self, idx=0, data_size=None, z=None):
        batch_size = self.batch_size if data_size is None else data_size
        if z is not None:
            self.z = z
        else:
            self.z = dict()
            self.z['z_inter'] = np.random.normal(0, 0.1, [batch_size, self.model.output_bar, self.model.z_inter_dim]).astype(np.float32)
        feed_dict = {self.model.z_inter: self.z['z_inter']}

        return feed_dict


