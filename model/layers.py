import tensorflow as tf
import numpy as np


def uvgg_pconv(name, input_layer, mask,
               concat_layer=None, concat_mask=None,
               n_in=None, n_out=None, weights=None,
               is_bnorm=False, is_train=False, is_reLU=True,
               is_trainable=True):
    """
    If weights exist, weights[0] are the kernel, weights[1] are the biases
    """
    with tf.variable_scope(name) as scope:
        if weights is None:
            kernel = \
                tf.get_variable('W', [3, 3, n_in, n_out], dtype=tf.float32,
                                initializer=tf.variance_scaling_initializer(),
                                trainable=is_trainable)
            biases = tf.get_variable('b', [n_out], dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=is_trainable)
        else:
            kernel = tf.get_variable(
                'W', dtype=tf.float32, initializer=tf.constant(weights[0]),
                trainable=is_trainable)
            biases = tf.get_variable(
                'b', dtype=tf.float32, initializer=tf.constant(weights[1]),
                trainable=is_trainable)

        # mask the input
        input_layer = tf.multiply(input_layer, mask)

        # mask kernel
        if concat_layer is None:
            kernel_mask = tf.ones([3, 3, n_in, n_out],
                                  dtype=tf.float32,
                                  name=None)
            mask_conv = tf.nn.conv2d(mask, kernel_mask,
                                     [1, 1, 1, 1], padding='SAME')
            mask_conv_w = tf.reciprocal(mask_conv+1e-7)
        else:
            # mask the concat
            concat_layer = tf.multiply(concat_layer, concat_mask)

            # and combine
            input_layer = tf.concat(
                axis=3, values=[concat_layer, input_layer], name='concat')

        conv = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')

        if concat_layer is None:
            # only encoder feautres
            conv = tf.multiply(conv, mask_conv_w)
            mask_conv = tf.cast(mask_conv > 0, dtype=tf.float32)
            conv = tf.multiply(conv, mask_conv)
            mask_out = mask_conv
        else:
            mask_out = mask
        out = tf.nn.bias_add(conv, biases)
        if is_bnorm:
            out = tf.layers.batch_normalization(
                out, renorm=True, training=False, trainable=False)
        if is_reLU:
            out = tf.nn.relu(out, 'act')
    return out, mask_out
    # if concat_layer is None else out, mask_out


def uvgg_conv(name, input_layer, concat_layer=None, n_in=None, n_out=None,
              weights=None, is_bnorm=False, is_train=False, is_reLU=True,
              is_trainable=True):
    """
    If weights exist, weights[0] are the kernel, weights[1] are the biases
    """
    with tf.variable_scope(name) as scope:
        if weights is None:
            kernel = \
                tf.get_variable('W', [3, 3, n_in, n_out], dtype=tf.float32,
                                initializer=tf.variance_scaling_initializer(),
                                trainable=is_trainable)
            biases = tf.get_variable('b', [n_out], dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=is_trainable)
        else:
            kernel = tf.get_variable(
                'W', dtype=tf.float32, initializer=tf.constant(weights[0]),
                trainable=is_trainable)
            biases = tf.get_variable(
                'b', dtype=tf.float32, initializer=tf.constant(weights[1]),
                trainable=is_trainable)
        if concat_layer is not None:
            input_layer = tf.concat(
                axis=3, values=[concat_layer, input_layer], name='concat')
        conv = tf.nn.conv2d(input_layer, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, biases)
        if is_bnorm:
            out = tf.layers.batch_normalization(
                out, renorm=True, training=False, trainable=False)
        if is_reLU:
            out = tf.nn.relu(out, name='act')
        else:
            print('no relu')
            # out = tf.nn.leaky_relu(out, name='act')
            # out = (tf.nn.sigmoid(out, name='act')-0.5)*255
    return out


def uvgg_pool(namepool, input_layer):
    """max pool wrapper"""
    # with tf.device('/cpu:0'):
    out = tf.layers.max_pooling2d(inputs=input_layer,
                                  pool_size=[2, 2],
                                  strides=2,
                                  padding="same")
    #     out = tf.nn.max_pool(input_layer,
    #                          ksize=[1, 2, 2, 1],
    #                          strides=[1, 2, 2, 1],
    #                          data_format="NHWC",
    #                          padding='SAME',
    #                          name=namepool)
    # return out
    return out


def uvgg_upmask(mask):
    # repeat rows and cols of input
    return tf.keras.layers.UpSampling2D(size=2,
                                        data_format="channels_last")(mask)


def uvgg_upconv(name, input_layer, n_in=None, n_out=None,
                up_factor=2, is_bnorm=False, is_train=False,
                is_trainable=True, is_pcon=False):
    """Things will be different.."""
    nb = tf.shape(input_layer)[0]
    n_wh = input_layer.get_shape().as_list()[1]*up_factor
    weights = bilinear_upsample_weights(2, n_out, n_in)
    # n_batch = input_layer.shape[0]
    with tf.variable_scope(name) as scope:
        if weights is None:
            kernel = \
                tf.get_variable('W', [4, 4, n_out, n_in], dtype=tf.float32,
                                initializer=tf.variance_scaling_initializer())
        else:
            kernel = tf.get_variable(
                'W', dtype=tf.float32, initializer=tf.constant(weights))
        conv = tf.nn.conv2d_transpose(input_layer,
                                      kernel,
                                      output_shape=[nb, n_wh, n_wh, n_out],
                                      strides=[1, 2, 2, 1])
        if is_pcon:
            conv = tf.multiply(conv, tf.reciprocal(tf.cast(27, tf.float32)))
        biases = tf.get_variable('b', [n_out], dtype=tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 trainable=is_trainable)
        out = tf.nn.bias_add(conv, biases)
        if is_bnorm:
            out = tf.layers.batch_normalization(
                out, renorm=True, training=False, trainable=False)
        out = tf.nn.relu(out)
    return out


def bilinear_upsample_weights(factor, no_out, no_chans):
    """
    Create weights matrix for transposed convolution with bilinear init.
    """
    def get_kernel_size(factor):
        """Find the kernel size given the desired factor of upsampling."""
        return 2 * factor - factor % 2

    def upsample_filt(size):
        """
        Make a 2D bilinear kernel for upsampling of the given (h, w) size.
        """
        # make filter
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
    # filtersize
    filter_size = get_kernel_size(factor)
    weights = np.zeros((filter_size,
                        filter_size,
                        no_out,
                        no_chans), dtype=np.float32)
    # get single filter
    upsample_kernel = upsample_filt(filter_size)
    for i in range(no_out):
        for j in range(no_chans):
            weights[:, :, i, j] = upsample_kernel
    return weights
