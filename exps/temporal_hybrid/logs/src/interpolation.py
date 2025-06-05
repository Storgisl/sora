import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pprint import pprint
import imageio

from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from input_data import InputDataNowBarHybrid
from config import TrainingConfig, OneBarHybridConfig
from musegan.libs.utils import slerp, bilerp, make_gif

# === GPU Setup (TF 2.x) ===
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# === Setup Model and Input ===
t_config = TrainingConfig
t_config.exp_name = 'exps/onebar_hybrid'
model = NowbarHybrid(OneBarHybridConfig)
input_data = InputDataNowBarHybrid(model)

# === Create MuseGAN model ===
musegan = MuseGAN(t_config=t_config, model=model)
musegan.build(input_shape=input_data.input_shape)
musegan.load_weights(musegan.dir_ckpt)

z_interpolation = dict()

# === Bilerp Interpolation (Grid) ===
gen_dir = 'interpolation/gen/bilerp'
os.makedirs(gen_dir, exist_ok=True)

inter_a0 = np.ones([64], dtype=np.float32) * -0.0005
intra_b0 = np.ones([64, 5], dtype=np.float32) * -0.0005
inter_a1 = np.ones([64], dtype=np.float32) * 0.0005
intra_b1 = np.ones([64, 5], dtype=np.float32) * 0.0005

grid_list = bilerp(inter_a0, inter_a1, intra_b0, intra_b1, 8)
z_interpolation['inter'] = np.array([t[0] for t in grid_list])
z_interpolation['intra'] = np.array([t[1] for t in grid_list])

# === Generate Samples ===
result, eval_result = musegan.gen_test(
    input_data,
    is_eval=True,
    gen_dir=gen_dir,
    key=None,
    is_save=True,
    z=z_interpolation,
    type_=1
)

# === Create GIF from Generated Images ===
make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir=gen_dir)

