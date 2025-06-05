import argparse
from musegan.core import MuseGAN
from musegan.components import TemporalHybrid
from input_data import InputDataTemporalHybrid
from config import TrainingConfig, TemporalHybridConfig
import tensorflow.compat.v1 as tf
import SharedArray as sa
import numpy as np
tf.disable_v2_behavior()

def ensure_shared_array_exists(name, shape=(100, 96, 84, 5), dtype=np.float32):
    try:
        sa.attach(name)
    except FileNotFoundError:
        shared_arr = sa.create(name, shape, dtype)
        shared_arr[:] = np.random.rand(*shape).astype(dtype)

parser = argparse.ArgumentParser()
parser.add_argument('--genre', type=str, default='classical')
parser.add_argument('--tempo', type=int, default=120)
parser.add_argument('--length_bars', type=int, default=4)
parser.add_argument('--instruments', type=str, default='piano,violin')

args = parser.parse_args()

t_config = TrainingConfig()
t_config.exp_name = 'exps/temporal_hybrid'
path_x_train_phr = 'tra_X_phrase_all'

ensure_shared_array_exists(path_x_train_phr)
model = TemporalHybrid(TemporalHybridConfig())
input_data = InputDataTemporalHybrid(model)
input_data.add_data_sa(path_x_train_phr, 'train')

with tf.Session(config=tf.ConfigProto()) as sess:
    musegan = MuseGAN(sess, t_config, model)
    musegan.load(musegan.dir_ckpt)

    config = {
        'genre': args.genre,
        'tempo': args.tempo,
        'length_bars': args.length_bars,
        'instruments': args.instruments.split(',')
    }

    musegan.generate_custom(input_data, config)

