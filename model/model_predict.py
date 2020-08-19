from tensorflow.keras import applications
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Concatenate, LeakyReLU, ReLU, PReLU
from tensorflow.keras.initializers import Constant, Zeros
from model.pconv_layer import PConv2D
from model.losses import loss_fn
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import numpy as np
from model.instance_layer import InstanceNormalization
import pdb
from model.losses import ssim_loss
from tensorflow.keras.regularizers import l1
import tensorflow as tf


def get_vgg(nrows, ncols, out_layer=None, is_trainable=False):
    # inputs = Input(shape=(nrows, ncols, 3))
    # make vgg percept up to pool 3
    vgg_model = applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    # Getting output tensor of the last VGG layer that we want to include
    #vgg_model.outputs = [layer_dict[out].output for out in out_layer]
    #
    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]

    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = False
    model.compile(loss='mse', optimizer='adam')
    return model

# def modify_vgg(nrows, ncols, out_layer=['block1_pool',
#                                      'block2_pool',
#                                      'block3_pool'], is_trainable=False):
#       """ bla bla """
#       for i_conv in range(nconv):
#         if weights:
#             nets, masks = PConv2D(nfilts, kernel_size,
#                                   activation=None,
#                                   padding='same',
#                                   use_bias=True,
#                                   kernel_initializer=Constant(
#                                       weights[2*i_conv+0]),
#                                   bias_initializer=Constant(
#                                       weights[2*i_conv+1]),
#                                   name=bname+'_conv_'+str(i_conv+1)
#                                   )([nets, masks])
#             if use_inst:
#                 nets = InstanceNormalization(axis=3)(nets)
#                 nets = LeakyReLU(alpha=0.2)(nets)
#             else:
#                 nets = ReLU()(nets)
#         else:
#             nets, masks = PConv2D(nfilts, kernel_size,
#                                   activation=None,
#                                   padding='same',
#                                   use_bias=True,
#                                   # activity_regularizer=l1(0.001),
#                                   bias_initializer='zeros',
#                                   kernel_initializer='he_normal',
#                                   name=bname+'_conv_'+str(i_conv+1)
#                                   )([nets, masks])
#             if use_inst:
#                 nets = InstanceNormalization(axis=3)(nets)
#             nets = LeakyReLU(alpha=0.2)(nets)
#             # nets = ReLU()(nets)
#     cnets = nets
#     cmasks = masks
#     # nets = MaxPooling2D((2, 2), strides=(2, 2), name='pooln_'+bname)(nets)
#     masks = MaxPooling2D((2, 2), strides=(
#         2, 2), name='poolm_'+bname)(masks)
#     nets = AveragePooling2D((2, 2), strides=(2, 2),
#                             name='pooln_'+bname)(nets)
#     # masks = AveragePooling2D((2, 2), strides=(2, 2), name='poolm_'+bname)(masks)
#     # return nets, masks, cnets, cmasks
#     # Create model and compile
#     model = Model(inputs=inputs, outputs=vgg_model(inputs))
#     model.trainable = False
#     model.compile(loss='mse', optimizer='adam')
#     return model


def make_pconv_uvgg(nets_in,
                    masks_in,
                    is_training=False,
                    vgg_model=None,
                    use_inst=False):

    # SUBFUNCTIONS
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

    # MAIN
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
        nets, masks, cnets_1, cmasks_1 = encoder_partial_vgg(
            nets_in, masks_in, 64, 2, 'vblock1', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_2, cmasks_2 = encoder_partial_vgg(
            nets, masks, 128, 2, 'vblock2', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_3, cmasks_3 = encoder_partial_vgg(
            nets, masks, 256, 3, 'vblock3', kernel_size=3, use_inst=use_inst)
        nets, masks, cnets_4, cmasks_4 = encoder_partial_vgg(
            nets, masks, 512, 3, 'vblock4', kernel_size=3, use_inst=use_inst)

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
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_5, cmasks_5, 512, 1, 'vblock6', use_inst=use_inst)    
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_4, cmasks_4, 512, 1, 'vblock7', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_3, cmasks_3, 256, 1, 'vblock8', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_2, cmasks_2, 128, 1, 'vblock9', use_inst=use_inst)
    nets, masks = decoder_partial_vgg(
        nets, masks, cnets_1, cmasks_1, 64, 1, 'vblock10', use_inst=use_inst)

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


def build_compile_model(mode, params):

    # load labels
    is_training = (mode == 'train')
    if is_training:
        nets_in = Input(shape=(224, 224, 3))
        masks_in = Input(shape=(224, 224, 3))
    else:
        nr = params['input_size']
        nets_in = Input(shape=(nr , nr, 3))
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
