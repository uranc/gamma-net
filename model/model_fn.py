from tensorflow.keras import applications
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Concatenate, LeakyReLU, ReLU, PReLU
from tensorflow.keras.initializers import Constant, Zeros
from model.pconv_layer import PConv2D, Conv2D
from model.losses import loss_fn, loss_fn_pred
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
from model.instance_layer import InstanceNormalization
import pdb
from model.losses import ssim_loss
from tensorflow.keras.regularizers import l1, l2
import tensorflow as tf
from keras import losses
from tensorflow.keras.layers import Dropout


def get_vgg(nrows, ncols, out_layer=None, is_trainable=False, inc_top=False):
    # inputs = Input(shape=(nrows, ncols, 3))
    # make vgg percept up to pool 3
    vgg_model = applications.VGG16(
        weights="imagenet",
        include_top=inc_top,
        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    # Getting output tensor of the last VGG layer that we want to include
    #vgg_model.outputs = [layer_dict[out].output for out in out_layer]
    for out in layer_dict.keys():
        # print('hello')
        layer_dict[out].trainable = is_trainable

    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]
        # layer_dict[out_layer[0]].trainable = True
    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = is_trainable
    model.compile(loss='mse', optimizer='adam')
    return model

def encoder_partial_vgg(nets, masks, nfilts, nconv, bname, kernel_size=3, weights=None, use_inst=False):
    for i_conv in range(nconv):
        if weights:
            nets, masks = PConv2D(nfilts, kernel_size,
                                  activation=None,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=Constant(
                                      weights[2*i_conv+0]),
                                  bias_initializer=Constant(
                                      weights[2*i_conv+1]),
                                  name=bname+'_conv_'+str(i_conv+1),
                                  # trainable=False
                                  )([nets, masks])
            if use_inst:
                nets = InstanceNormalization(axis=3)(nets)
                # nets = LeakyReLU(alpha=0.2)(nets)
                nets = ReLU()(nets)
            else:
                nets = ReLU()(nets)
                # nets = PReLU()(nets)
        else:
            nets, masks = PConv2D(nfilts, kernel_size,
                                  activation=None,
                                  padding='same',
                                  use_bias=True,
                                  # activity_regularizer=l1(0.001),
                                  bias_initializer='zeros',
                                  kernel_initializer='he_normal',
                                  name=bname+'_conv_'+str(i_conv+1)
                                  )([nets, masks])
            if use_inst:
                nets = InstanceNormalization(axis=3)(nets)
            # nets = LeakyReLU(alpha=0.2)(nets)
            # nets = PReLU()(nets)
            nets = ReLU()(nets)
    cnets = nets
    cmasks = masks
    # nets = MaxPooling2D((2, 2), strides=(2, 2), name='pooln_'+bname)(nets)
    if not bname == 'vblock5':
        masks = MaxPooling2D((2, 2), strides=(
            2, 2), name='poolm_'+bname)(masks)
        nets = MaxPooling2D((2, 2), strides=(2, 2),
                            name='pooln_'+bname)(nets)
    # masks = AveragePooling2D((2, 2), strides=(2, 2), name='poolm_'+bname)(masks)
    return nets, masks, cnets, cmasks


def decoder_partial_vgg(nets, masks,
                        cnets, cmasks,
                        nfilts, nconv, bname, use_inst=False):

    # upsample
    # init_weights = bilinear_upsample_weights(
    #     2, nfilts, nets.get_shape().as_list()[-1])
    # nets = Conv2DTranspose(nfilts, (4, 4),
    #                        strides=(2, 2),
    #                        padding='same',
    #                        kernel_initializer=Constant(init_weights),
    #                        name=bname+'_upconv')(nets)

    # # upsample masks
    nets = UpSampling2D(size=(2, 2))(nets)
    masks = UpSampling2D(size=(2, 2))(masks)
    # nets, masks = PConv2D(nfilts, 3,
    #                       activation=None,
    #                       padding='same',
    #                       use_bias=True,
    #                       # activity_regularizer=l1(0.001),
    #                       bias_initializer='zeros',
    #                       kernel_initializer='he_normal',
    #                       name=bname+'_upconv',
    #                       )([nets, masks])
    # if use_inst:
    #     nets = InstanceNormalization(axis=3)(nets)
    # # nets = LeakyReLU(alpha=0.2)(nets)
    # nets = ReLU()(nets)

    # nets = PReLU()(nets)
    # def get_sliced_shape(x):
    #     shape_list = x.get_shape().as_list()
    #     return x[:, :, :, :int(shape_list[-1]/2)]
    # masks = Lambda(get_sliced_shape)(masks)

    # concat
    nets = Concatenate(axis=-1)([nets, cnets])
    masks = Concatenate(axis=-1)([masks, cmasks])

    # # 2nd conv
    for i_conv in range(nconv):
        nets, masks = PConv2D(nfilts, 3,
                              activation=None,
                              padding='same',
                              use_bias=True,
                              # activity_regularizer=l1(0.001),
                              bias_initializer='zeros',
                              kernel_initializer='he_normal',
                              name=bname+'_conv_'+str(i_conv+1)
                              )([nets, masks])
        if use_inst:
            nets = InstanceNormalization(axis=3)(nets)
        # nets = LeakyReLU(alpha=0.2)(nets)
        nets = ReLU()(nets)
        # nets = PReLU()(nets)
    # refine perhaps
    return nets, masks

def encoder_dummy_vgg(nets, nfilts, nconv, bname, kernel_size=3, weights=None, use_inst=False):
    for i_conv in range(nconv):
        nets = Conv2D(nfilts, kernel_size,
                      activation=None,
                      padding='same',
                      use_bias=True,
                      # activity_regularizer=l1(0.001),
                      bias_initializer='zeros',
                      kernel_initializer='he_normal',
                      name=bname+'_conv_'+str(i_conv+1)
                      )(nets)
    cnets = nets
    # nets = MaxPooling2D((2, 2), strides=(2, 2), name='pooln_'+bname)(nets)
    if not bname == 'vblock5':
        nets = MaxPooling2D((2, 2), strides=(2, 2),
                            name='pooln_'+bname)(nets)
    # masks = AveragePooling2D((2, 2), strides=(2, 2), name='poolm_'+bname)(masks)
    return nets, cnets

# def decoder_partial_vgg(nets, nfilts, nconv, bname, use_inst=False):
#     # upsample masks
#     # nets = UpSampling2D(size=(2, 2))(nets)
#     nets = Conv2D(nfilts, 2,
#                   activation=None,
#                   padding='same',
#                   use_bias=True,
#                   # activity_regularizer=l1(0.001),
#                   bias_initializer='zeros',
#                   kernel_initializer='he_normal',
#                   name=bname+'_conv_T'
#                   )(nets)
#     # concat
#     # nets = Concatenate(axis=-1)(nets)

#     # # 2nd conv
#     for i_conv in range(nconv):
#         nets = Conv2D(nfilts, 3,
#                       activation=None,
#                       padding='same',
#                       use_bias=True,
#                       # activity_regularizer=l1(0.001),
#                       bias_initializer='zeros',
#                       kernel_initializer='he_normal',
#                       name=bname+'_conv_'+str(i_conv+1)
#                       )(nets)
#         nets = ReLU()(nets)
#     return nets

def make_uvgg(nets_in,
              is_features=False,
              is_training=False,
              vgg_model=None,
              use_inst=False):

    use_inst = False
    nets, cnets_1 = encoder_dummy_vgg(nets_in, 64, 2, 'vblock1', kernel_size=3, use_inst=use_inst)
    nets, cnets_2 = encoder_dummy_vgg(nets, 128, 2, 'vblock2', kernel_size=3, use_inst=use_inst)
    nets, cnets_3 = encoder_dummy_vgg(nets, 256, 3, 'vblock3', kernel_size=3, use_inst=use_inst)
    nets, cnets_4 = encoder_dummy_vgg(nets, 512, 3, 'vblock4', kernel_size=3, use_inst=use_inst)
    nets, cnets_5 = encoder_dummy_vgg(nets, 512, 3, 'vblock5', kernel_size=3, use_inst=use_inst)
    
    nets = decoder_partial_vgg(nets, 512, 2, 'vblock7', use_inst=use_inst)
    nets = decoder_partial_vgg(nets, 256, 2, 'vblock8', use_inst=use_inst)
    nets = decoder_partial_vgg(nets, 128, 2, 'vblock9', use_inst=use_inst)
    nets = decoder_partial_vgg(nets, 64, 2, 'vblock10', use_inst=use_inst)

    # output
    nets = Conv2D(3, (3, 3),
                  activation=None,
                  padding='same',
                  use_bias=True,
                  # activity_regularizer=l1(0.001),
                  bias_initializer='zeros',
                  kernel_initializer='he_normal',
                  name='block10')(nets)
    return nets

def make_pconv_uvgg(nets_in,
                    masks_in,
                    is_features=False,
                    is_training=False,
                    vgg_model=None,
                    use_inst=False):
    # encoder
    if not use_inst:
        weights = vgg_model.get_weights()
        nets, masks, cnets_1, cmasks_1 = encoder_partial_vgg(
            nets_in, masks_in, 64, 2, 'vblock1', weights=weights[0*2:2*2], use_inst=False)
        nets, masks, cnets_2, cmasks_2 = encoder_partial_vgg(
            nets, masks, 128, 2, 'vblock2', weights=weights[2*2:4*2], use_inst=False)
        nets, masks, cnets_3, cmasks_3 = encoder_partial_vgg(
            nets, masks, 256, 3, 'vblock3', weights=weights[4*2:7*2], use_inst=False)
        nets, masks, cnets_4, cmasks_4 = encoder_partial_vgg(
            nets, masks, 512, 3, 'vblock4', weights=weights[7*2:10*2], use_inst=False)
        nets, masks, cnets_5, cmasks_5 = encoder_partial_vgg(
            nets, masks, 512, 3, 'vblock5', weights=weights[10*2:13*2], use_inst=False)
    else:
        use_inst = False
        nets, masks, cnets_1, cmasks_1 = encoder_partial_vgg(
            nets_in, masks_in, 64, 2, 'vblock1', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_2, cmasks_2 = encoder_partial_vgg(
            nets, masks, 128, 2, 'vblock2', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_3, cmasks_3 = encoder_partial_vgg(
            nets, masks, 256, 3, 'vblock3', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_4, cmasks_4 = encoder_partial_vgg(
            nets, masks, 512, 3, 'vblock4', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_5, cmasks_5 = encoder_partial_vgg(
            nets, masks, 512, 3, 'vblock5', kernel_size=3, use_inst=use_inst)
    # bottleneck
    # nets, masks = PConv2D(512, (3, 3),
    #                       activation=None,
    #                       padding='same',
    #                       use_bias=True,
    #                       # activity_regularizer=l1(0.001),
    #                       bias_initializer='zeros',
    #                       kernel_initializer='he_normal',
    #                       name='vblock5'+'_conv_1')([nets, masks])
    # if use_inst:
    #     nets = InstanceNormalization(axis=3)(nets)
    # # nets = LeakyReLU(alpha=0.2)(nets)
    # nets = ReLU()(nets)
    # # nets = PReLU()(nets)
    
    # decoder
    # nets, masks = decoder_partial_vgg(
    #     nets, masks, cnets_5, cmasks_5, 512, 1, 'vblock6', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_4, cmasks_4, 512, 2, 'vblock7', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_3, cmasks_3, 256, 2, 'vblock8', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_2, cmasks_2, 128, 2, 'vblock9', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_1, cmasks_1, 64, 2, 'vblock10', use_inst=use_inst)

    # pdb.set_trace()
    # output
    nets = Concatenate(axis=-1)([nets, nets_in])
    masks = Concatenate(axis=-1)([masks, masks_in])
    nets, masks = PConv2D(3, (3, 3),
                          activation=None,
                          padding='same',
                          use_bias=True,
                          # activity_regularizer=l1(0.001),
                          bias_initializer='zeros',
                          kernel_initializer='he_normal',
                          name='block10')([nets, masks])
    return nets, masks

def build_compile_empty_model(mode, params):

    # load labels
    is_training = (mode == 'train')
    if is_training:
        nets_in = Input(shape=(224, 224, 3))
    else:
        nr = params['input_size']
        nets_in = Input(shape=(nr, nr, 3))

    # Settings
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # Create UNet-like model
    outputs = make_uvgg(nets_in,
                         vgg_model=None,
                         is_training=is_training,
                         use_inst=params['inst_norm'])

    model = Model(inputs=[nets_in], outputs=[outputs])
    # compile
    model.compile(optimizer='adam', loss='mse')
    return model

def build_compile_model(mode, params):

    # load labels
    is_training = (mode == 'train')
    if is_training:
        nets_in = Input(shape=(224, 224, 3))
        masks_in = Input(shape=(224, 224, 3))
    else:
        nr = params['input_size']
        nets_in = Input(shape=(nr, nr, 3))
        masks_in = Input(shape=(nr, nr, 3))

    # Settings
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # get vgg16 layers pool1, pool2, pool3
    vgg_model = get_vgg(nrows, ncols, out_layer=['block1_conv2',
                                                 'block2_conv2',
                                                 'block3_conv3',
                                                 'block4_conv3',
                                                 ])
    # 'block2_pool',
    # 'block3_pool'
    # Create UNet-like model
    outputs, masks = make_pconv_uvgg(nets_in,
                                     masks_in,
                                     vgg_model=vgg_model,
                                     is_training=is_training,
                                     use_inst=params['inst_norm'])

    # Setup the model inputs / outputs
    # output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    model = Model(inputs=[nets_in]+[masks_in], outputs=[outputs])

    # if not is_training:
    #     all_out = []
    #     for layer in model.layers:
    #         if (layer.name[:5] == 're_lu') or (layer.name[:5] == 'pooln') or (layer.name[:5] == 'block10'):
    #             all_out.append(layer)
    #     model = Model(inputs=[nets_in]+[masks_in], outputs=[all_out])

    # model.mode = mode
    total_loss = loss_fn(params, masks_in, vgg_model=vgg_model)
    ssim = ssim_loss(masks_in)

    # load weights i
    if params['weight_dir']:
        print('load')
        model.load_weights(params['weight_dir'])

    # compile
    model.compile(optimizer=optimizers.Adam(lr=params['lr']),
                  loss=total_loss, metrics=[ssim])
    return model


def build_compile_model_pred(mode, params):

    # load labels
    is_training = (mode == 'train')
    if is_training:
        nets_in = Input(shape=(84, 84, 3))
        # nets_in = Input(shape=(224, 224, 3))
    else:
        nets_in = Input(shape=(84, 84, 3))
        # nets_in = Input(shape=(224, 224, 3))
        # nr = params['input_size']
        # nets_in = Input(shape=(nr, nr, 3))

    # Settings
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # get vgg16 layers pool1, pool2, pool3
    if False:
        vgg_model = get_vgg(nrows, ncols, out_layer=['fc2',
                                                     ],
                            is_trainable=False,
                            inc_top=True)
        #'block1_conv2', 'block2_conv2',
        nets = vgg_model(nets_in)
        nets = Dense(1024, activation=None,
                     bias_initializer='zeros',
                     kernel_initializer='he_normal')(nets)
        nets = InstanceNormalization(axis=3)(nets)
        nets = ReLU()(nets)
        outputs = Dense(1, activation=None,
                        bias_initializer='zeros',
                        kernel_initializer='he_normal')(nets)
    else:
        vgg_model = get_vgg(nrows, ncols,  # block3_conv3
                            out_layer=['block3_conv3'],
                            is_trainable=False)
        all_out = []
        #'block1_conv2', 'block2_conv2',
        vgg_outs = vgg_model(nets_in)
        # o1 = Flatten()(vgg_outs[0])block3_conv3
        # o2 = Flatten()(vgg_outs[1])
        # nets = Concatenate(axis=-1)([vgg_outs[0], vgg_outs[1]])
        nets = vgg_outs  # Concatenate(axis=-1)([vgg_outs[0]])
        all_out.append(nets)

        # -1
        # nets = Conv2D(128, (3, 3),
        #               activation=None,
        #               padding='valid',
        #               use_bias=True,
        #               strides=2,
        #               kernel_regularizer=l1(0.001),
        #               bias_initializer='zeros',
        #               kernel_initializer='he_normal',
        #               name='squeeze_00')(nets)
        # all_out.append(nets)
        # nets = LeakyReLU(0.1)(nets)
        # nets = Dropout(0.5)(nets)
        # all_out.append(nets) 
        # 0
        # nets = Conv2D(64, (3, 3),
        #               activation=None,
        #               padding='valid',
        #               use_bias=True,
        #               strides=2,
        #               kernel_regularizer=l1(0.001),
        #               bias_initializer='zeros',
        #               kernel_initializer='he_normal',
        #               name='squeeze_0')(nets)
        # all_out.append(nets)
        # nets = LeakyReLU(0.1)(nets)
        # nets = Dropout(0.5)(nets)
        # all_out.append(nets) 
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
        all_out.append(nets)
        # nets = LeakyReLU(0.1)(nets)
        nets = Dropout(0.5)(nets)
        all_out.append(nets)
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
        all_out.append(nets)
        # nets = LeakyReLU(0.1)(nets)
        nets = Dropout(0.5)(nets)
        all_out.append(nets)
        # 3
        # nets = Conv2D(16, (3, 3),
        #               activation=None,
        #               padding='valid',
        #               use_bias=True,
        #               # strides=2,
        #               kernel_regularizer=l1(0.001),
        #               bias_initializer='zeros',
        #               kernel_initializer='he_normal',
        #               name='squeeze_3')(nets)
        # all_out.append(nets)                      
        # nets = LeakyReLU(0.1)(nets)
        # nets = Dropout(0.5)(nets)
        # all_out.append(nets)
        # nets = Conv2D(32, (3, 3),
        #               activation=None,
        #               padding='valid',
        #               use_bias=True,
        #               strides=2,
        #               kernel_regularizer=l1(0.001),
        #               bias_initializer='zeros',
        #               kernel_initializer='he_normal',
        #               name='squeeze_4')(nets)
        # nets = LeakyReLU(0.1)(nets)
        # # nets = ReLU()(nets)
        # print(nets.shape)
        # nets = Conv2D(1, (21, 21),
        #                    activation=None,
        #                    padding='valid',
        #                    use_bias=True,
        #                    # strides=10,
        #                 #    kernel_regularizer=l1(0.1),
        #                 #    kernel_regularizer=l1(0.001),
        #                    bias_initializer='zeros',
        #                    kernel_initializer='he_normal',
        #                    name='conv_out')(nets)        
        nets = Conv2D(1, (4, 4),
                           activation=None,
                           padding='valid',
                           use_bias=True,
                           # strides=10,
                           # kernel_regularizer=l1(0.001),
                           bias_initializer='zeros',
                           kernel_initializer='he_normal',
                           name='conv_out')(nets)
        all_out.append(nets)                           
        # nets = LeakyReLU(0.1)(nets)
        # all_out.append(nets)
        # outputs = nets_peak#*nets_sig
        if is_training:
            outputs = Flatten()(nets)
        else:
            nets = Flatten()(nets)
            all_out.append(nets)
            outputs = all_out
    
    # Setup the model inputs / outputs
    model = Model(inputs=[nets_in], outputs=[outputs])

    if params['weight_dir']:
        print('load')
        model.load_weights(params['weight_dir'])

    # compile
    model.compile(optimizer=optimizers.Adam(lr=params['lr']),
                  loss=losses.mean_squared_error)
    return model


# def build_compile_model_Upred(mode, params):
#     from tensorflow.keras import Input
#     # load labels
#     is_training = (mode == 'train')
#     # nets = vgg_outs  # Concatenate(axis=-1)([vgg_outs[0]])

#     import importlib
#     a_model = importlib.import_module('experiments.{0}.model.model_fn'.format('f4'))
#     import copy
#     tmp_params = copy.deepcopy(params)
#     tmp_params['weight_dir'] = 'experiments/f4/weights.last.h5'
#     model = a_model.build_compile_model(mode, tmp_params)
#     model.trainable = False
#     # model.summary()
#     from tensorflow.keras.models import Model
    
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])    
#     outputs = [layer_dict[out].output for ind, out in enumerate(layer_dict.keys()) if out[:5]=='re_lu' or out[:5]=='pooln']
#     outputs = outputs[8]
#     # nets_in = Input(shape=(224, 224, 3))
#     # masks_in = Input(shape=(224, 224, 3))
#     uprednet = Model(inputs=model.inputs, outputs=outputs)
#     # uprednet = Model(inputs=[nets_in]+[masks_in], outputs=outputs)
#     uprednet.trainable = False
#     uprednet.compile(loss='mse', optimizer='adam')
#     # uprednet.summary()
#     # nets_in = Input(shape=(112, 112, 3))
#     # masks_in = Input(shape=(112, 112, 3))    
#     nets_in = Input(shape=(84, 84, 3))
#     masks_in = Input(shape=(84, 84, 3))
#     nets = uprednet([nets_in]+[masks_in])
#     # 1
#     nets = Conv2D(256, (3, 3),
#                   activation=None,
#                   padding='valid',
#                   use_bias=True,
#                   strides=2,
#                   kernel_regularizer=l1(0.01),
#                   bias_initializer='zeros',
#                   kernel_initializer='he_normal',
#                   name='squeeze_1')(nets)
#     nets = LeakyReLU(0.1)(nets)
#     # nets = ReLU()(nets)
#     print(nets.shape)
#     # 2
#     nets = Conv2D(32, (3, 3),
#                   activation=None,
#                   padding='valid',
#                   use_bias=True,
#                   strides=2,
#                   kernel_regularizer=l1(0.01),
#                   bias_initializer='zeros',
#                   kernel_initializer='he_normal',
#                   name='squeeze_2')(nets)
#     nets = LeakyReLU(0.1)(nets)
#     # nets = ReLU()(nets)

#     nets_peak = Conv2D(1, (4, 4),
#                        activation=None,
#                        padding='valid',
#                        use_bias=True,
#                        kernel_regularizer=l1(0.01),
#                        bias_initializer='zeros',
#                        kernel_initializer='he_normal',
#                        name='conv_out')(nets)
#     # nets_peak = LeakyReLU(0.1)(nets_peak)
#     # nets_peak = ReLU()(nets_peak)
#     # nets_out = nets
#     # nets_out = nets_peak#*nets_sig
#     # outputs = Flatten()(nets_out)
#     outputs = nets_peak
#     # outputs = Dense(256, activation=None,
#     #              bias_initializer='zeros',
#     #              use_bias=True,
#     #              kernel_regularizer=l1(0.001),
#     #              kernel_initializer='he_normal')(outputs)
#     # outputs = LeakyReLU(0.1)(outputs)
#     # outputs = Dense(64, activation=None,
#     #                 bias_initializer='zeros',
#     #                 use_bias=True,
#     #                 # kernel_regularizer=l1(0.001),
#     #                 kernel_initializer='he_normal')(outputs)
#     # outputs = LeakyReLU(0.1)(outputs)
#     # outputs = Dense(1, activation=None,
#     #                 bias_initializer='zeros',
#     #                 use_bias=True,
#     #                 # kernel_regularizer=l1(0.001),
#     #                 kernel_initializer='he_normal')(outputs)
#     # outputs = LeakyReLU(0.1)(outputs)
#     outputs = Flatten()(outputs)
#     # nets_peak = ReLU()(nets_peak)
#     # nets_out = nets
#     # nets_out = nets_peak#*nets_sig
#     # outputs = Flatten()(nets_out)
#     model = Model(inputs=[nets_in]+[masks_in], outputs=[outputs])

#     if params['weight_dir']:
#         print('load')
#         model.load_weights(params['weight_dir'])

#     # compile
#     model.compile(optimizer=optimizers.Adam(lr=params['lr']),
#                   loss=losses.mean_squared_error)
#     return model
