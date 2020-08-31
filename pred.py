import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from model.model_fn import build_compile_model_pred

def get_numpy_dataset(fname, batch_size=64):
    tmp = np.transpose(np.load(fname), [3,0,1,2])
    BATCH_SIZE = batch_size
    test_dataset = tf.data.Dataset.from_tensor_slices((tmp, np.zeros((tmp.shape[0], 1))))
    test_dataset = test_dataset.batch(BATCH_SIZE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
    initializer = iterator.make_initializer(test_dataset)
    return (iterator, initializer), tmp.shape[0]


parser = argparse.ArgumentParser(
    description='gamma-net predicts log10 gamma power for a given image')
parser.add_argument('--input', type=str, nargs=1,
                    help='input size 84x84 .png or .npy', default='examples/sample.png')
args = parser.parse_args()
input_name = args.input[0]

out_name, file_ext = os.path.splitext(input_name)

# params
WEIGHT_DIR = 'weights.last.h5'
BATCH_SIZE = 64

# model
model = build_compile_model_pred(WEIGHT_DIR)
model.summary()    

# modelname
flag_numpy = 1 if file_ext=='.npy' else 0

if flag_numpy:
    out, test_size = get_numpy_dataset(input_name, BATCH_SIZE)
    test_steps = int(np.floor(test_size/BATCH_SIZE))+1
    test_inputs, initializer = out
    sess = tf.Session()
    im_in, lab = test_inputs.get_next()
    pred = model.predict(test_inputs.get_next(), steps=test_steps)
    np.save(out_name + '_pred.npy', pred)
else:
    from skimage.io import imread
    from skimage.transform import resize
    img = imread(input_name)

    # resize image to 84x84
    img = resize(img, (84, 84, 3), anti_aliasing=True)
    pred = model.predict(np.expand_dims(img, axis=0), steps=1)
    print(pred[0][0])



