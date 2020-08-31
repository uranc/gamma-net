from tensorflow.keras import applications
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Concatenate, LeakyReLU, ReLU, PReLU
from tensorflow.keras.initializers import Constant, Zeros
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf
from keras import losses
from tensorflow.keras.layers import Dropout


def get_vgg(nrows, ncols, out_layer=None, is_trainable=False, inc_top=False):

    # make vgg percept up to pool 3
    vgg_model = applications.VGG16(
        weights="imagenet",
        include_top=inc_top,
        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
   
    # Getting output tensor of the last VGG layer that we want to include
    for out in layer_dict.keys():
        layer_dict[out].trainable = is_trainable

    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]
        
    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = is_trainable
    model.compile(loss='mse', optimizer='adam')
    return model

def build_compile_model_pred(weight_dir):

    # load labels
    nets_in = Input(shape=(84, 84, 3))


    # Settings
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # get vgg16 layers pool1, pool2, pool3
    vgg_model = get_vgg(nrows, ncols,
                        out_layer=['block3_conv3'],
                        is_trainable=False)
    nets = vgg_model(nets_in)

    # 1
    nets = Conv2D(32, (3, 3),
                    activation=None,
                    padding='valid',
                    use_bias=True,
                    strides=2,
                    kernel_regularizer=l1(0.001),
                    bias_initializer='zeros',
                    kernel_initializer='he_normal',
                    name='squeeze_1')(nets)
    nets = LeakyReLU(0.1)(nets)
    nets = Dropout(0.5)(nets)

    # 2
    nets = Conv2D(16, (3, 3),
                    activation=None,
                    padding='valid',
                    use_bias=True,
                    strides=2,
                    kernel_regularizer=l1(0.001),
                    bias_initializer='zeros',
                    kernel_initializer='he_normal',
                    name='squeeze_2')(nets)
    nets = LeakyReLU(0.1)(nets)
    nets = Dropout(0.5)(nets)

    #    
    nets = Conv2D(1, (4, 4),
                        activation=None,
                        padding='valid',
                        use_bias=True,
                        bias_initializer='zeros',
                        kernel_initializer='he_normal',
                        name='conv_out')(nets)
    nets = Flatten()(nets)
    outputs = nets
    
    # Setup the model inputs / outputs
    model = Model(inputs=[nets_in], outputs=[outputs])

    if weight_dir:
        print('loading model')
        model.load_weights(weight_dir)

    # compile
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss=losses.mean_squared_error)
    return model
