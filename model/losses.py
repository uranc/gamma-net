from tensorflow.keras import backend as K
from tensorflow.keras.layers import Cropping2D
import tensorflow as tf
import numpy as np


def loss_fn_pred():
    """Loss function"""
    def loss(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        return K.sum(K.abs(y_pred-y_true))

def loss_fn(params, masks, vgg_model=None):
    """Loss function"""
    def loss(y_true, y_pred):
        def _linear_decorelate_color(t):
            """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.
            If you interpret t's innermost dimension as describing colors in a
            decorrelated version of the color space (which is a very natural way to
            describe colors -- see discussion in Feature Visualization article) the way
            to map back to normal colors is multiply the square root of your color
            correlations.
            """
            color_correlation_svd_sqrt = \
                np.linalg.inv(np.asarray([[0.27, -0.09, 0.03],
                                          [0.27, 0.00, -0.05],
                                          [0.26, 0.09, 0.02]]).astype("float32").T)
            # max_norm_svd_sqrt = np.max(np.linalg.norm(
            #     color_correlation_svd_sqrt, axis=0))
            # color_mean = [0.48, 0.46, 0.41]
            # check that inner dimension is 3?
            t_flat = tf.reshape(t, [-1, 3])
            # color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
            # color_correlation_normalized = color_correlation_svd_sqrt
            # t_flat = tf.matmul(t_flat, np.linalg.inv(
            #     color_correlation_svd_sqrt))
            t_flat = tf.matmul(t_flat, color_correlation_svd_sqrt)
            # t = tf.reshape(t_flat, tf.shape(t))
            t = tf.reshape(t_flat, tf.shape(t))
            return t
        # mean = K.constant([103.939, 116.779, 123.68],
        #               dtype=tf.float32,
        #               shape=[1, 1, 1, 3],
        #               name='img_mean')
        # p_pred = K.clip(y_pred+mean, 0, 255)
        # p_true = K.clip(y_true+mean, 0, 255)
        # pixel loss
        # pixel_val = l1_loss(y_pred, y_true)
        # pixel_val = l2_loss(y_pred, y_true)
        # p_pred = _linear_decorelate_color(p_pred[:,:,:,::-1])
        # p_true = _linear_decorelate_color(p_true[:,:,:,::-1])
        # collect loss
        # pixel_val = 0
        # total_loss = pixel_val
        total_loss = 0
        # total_loss += fft_loss(_linear_decorelate_color(y_pred),
        # _linear_decorelate_color(y_true))
        total_loss += fft_loss(y_pred, y_true, params)
        # total_loss += fft_loss(p_pred, p_true, params)
        # total_loss = 0
        if params['p_hole']:
            hole_val = mask_loss(y_pred, y_true, masks)
            total_loss += params['p_hole']*hole_val
        if params["p_content"]:
            vgg_y_pred = vgg_model(y_pred)
            vgg_y_true = vgg_model(y_true)
            vgg_val = content_loss(vgg_y_pred, vgg_y_true)
            total_loss += params["p_content"]*vgg_val
        # # # if params.p_ssim:
        # # #     ssim_val = ssim_loss(y_pred, y_true)
        # # #     total_loss += params.p_ssim*ssim_val
        if params['p_tv']:
            tv_val = tv_loss(y_pred, y_true)
            total_loss += params['p_tv']*tv_val
        # # return total_loss, pixel_val, hole_val, vgg_val, \
        #     # ssim_val, tv_val, style_val
        if params['p_fft_c']:
            vgg_y_pred = vgg_model(y_pred)
            vgg_y_true = vgg_model(y_true)
            fft_c_val = fft_content(vgg_y_pred, vgg_y_true, params)
            total_loss += params["p_fft_c"]*fft_c_val
        # if params["p_style"]:
        #     # vgg_y_pred = vgg_model(y_pred)
        #     # vgg_y_true = vgg_model(y_true)
        #     style_val = style_loss(vgg_y_pred, vgg_y_true)
        #     total_loss += params["p_style"]*style_val
        if params['p_fft_s']:
            # vgg_y_pred = vgg_model(y_pred)
            # vgg_y_true = vgg_model(y_true)
            fft_s_val = fft_style(vgg_y_pred, vgg_y_true, params)
            total_loss += params["p_fft_s"]*fft_s_val
        return total_loss
    return loss


def mask_loss(y_pred, y_true, masks):
    return fft_loss(tf.multiply(y_pred, 1-masks), tf.multiply(y_true, 1-masks))


def fft_loss(p_pred, p_true, params):
    ndim = len(p_pred.shape)
    batch, h, w, ch = p_pred.shape.as_list()
    fft_pred = tf.spectral.rfft2d(K.permute_dimensions(
        p_pred, (0, 3, 1, 2)))
    fft_true = tf.spectral.rfft2d(K.permute_dimensions(
        p_true, (0, 3, 1, 2)))

    # log magnitude
    fft_log_mag = (K.log(K.abs(fft_pred)+K.epsilon())
                   - K.log(K.abs(fft_true)+K.epsilon()))
    total = params['p_fft_log']*K.sum(K.mean(K.abs(fft_log_mag), axis=[1, 2, 3]))

    # mag
    fft_mag = K.abs(K.abs(fft_pred)-K.abs(fft_true))
    total += params['p_fft_abs']*K.sum(K.mean(K.abs(fft_mag), axis=[1, 2, 3]))
    # phase
    fft_phase = K.abs(fft_pred)*K.abs(fft_true) \
        - tf.real(fft_pred)*tf.real(fft_true) \
        - tf.imag(fft_pred)*tf.imag(fft_true)
    total += params['p_fft_phase']*K.sum(K.mean(K.abs(fft_phase), axis=[1, 2, 3]))
    return total


def fft_content(vgg_y_pred, vgg_y_true, params):
    content_val = 0
    for i_layer in range(len(vgg_y_pred)):
        content_val += fft_loss(vgg_y_pred[i_layer], vgg_y_true[i_layer], params)
    return content_val


def fft_style(vgg_y_pred, vgg_y_true, params):
    def fft_gram(x, ba, hi, wi, ch):
        # gram
        if ba is None:
            ba = -1
        feature = tf.reshape(x, [ba, int(hi * wi), ch])
        # print(feature.shape)
        gram = K.batch_dot(feature, feature, axes=1)
        # print(gram.shape)
        return gram / tf.cast(hi * wi * ch, x.dtype)
    # style
    style_val = 0
    for i_layer in range(len(vgg_y_pred)):
        # fft the layers
        fft_s_pred = K.abs(tf.spectral.rfft2d(
            K.permute_dimensions(vgg_y_pred[i_layer], (0, 3, 1, 2))))
        fft_s_true = K.abs(tf.spectral.rfft2d(
            K.permute_dimensions(vgg_y_true[i_layer], (0, 3, 1, 2))))
        ba, ch, hi, wi = fft_s_pred.get_shape().as_list()
        style_val += l1_loss(fft_gram(fft_s_pred, ba, hi, wi, ch),
                             fft_gram(fft_s_true, ba, hi, wi, ch))
    return style_val


def l2_loss(y_pred, y_true):
    ndim = len(y_pred.shape)
    # print(ndim)
    if ndim == 4:
        total = K.sum(K.mean(K.square(y_pred-y_true), axis=[1, 2, 3]))
    elif ndim == 3:
        total = K.sum(K.mean(K.square(y_pred-y_true), axis=[1, 2]))
    return total


def l1_loss(y_pred, y_true):
    ndim = len(y_pred.shape)
    # print(ndim)
    # print(y_pred)
    if ndim == 4:
        total = K.sum(K.mean(K.abs(y_pred-y_true), axis=[1, 2, 3]))
    elif ndim == 3:
        total = K.sum(K.mean(K.abs(y_pred-y_true), axis=[1, 2]))
    # print(total)
    return total


def content_loss(vgg_y_pred, vgg_y_true):
    # define perceptual loss based on VGG16
    vgg_val = 0
    for i_layer in range(len(vgg_y_pred)):
        vgg_val += l2_loss(vgg_y_pred[i_layer], vgg_y_true[i_layer])
    return vgg_val


def style_loss(vgg_y_pred, vgg_y_true):
    style_val = 0
    # if vgg_y_pred is not None and vgg_y_true is not None:
    for i_layer in range(len(vgg_y_pred)):
        ba, hi, wi, ch = vgg_y_pred[i_layer].get_shape().as_list()
        gram_y_pred = gram_matrix(vgg_y_pred[i_layer], ba, hi, wi, ch)
        gram_y_true = gram_matrix(vgg_y_true[i_layer], ba, hi, wi, ch)
        style_val += l2_loss(gram_y_pred, gram_y_true)
    return style_val


def gram_matrix(x, ba, hi, wi, ch):
    """gram for input"""
    if ba is None:
        ba = -1
    feature = K.reshape(x, [ba, int(hi * wi), ch])
    gram = K.batch_dot(feature, feature, axes=1)
    return gram / (hi * wi * ch)

# def tv_loss(y_pred, y_true):
#     # tv norm
#     # c_pred = Cropping2D(cropping=((40, 40), (40, 40)))(y_pred)
#     c_pred = Cropping2D(cropping=((44, 44), (44, 44)))(y_pred)
#     mean = K.constant([103.939, 116.779, 123.68],
#                       dtype=tf.float32,
#                       shape=[1, 1, 1, 3],
#                       name='img_mean')
#     c_pred = K.clip(c_pred+mean, 0, 255)
#     return K.sum(tf.image.total_variation(y_pred))
#     # return K.sum(tf.image.total_variation(y_pred)-tf.image.total_variation(y_true), axis=[0])

def tv_loss(y_pred, y_true):
    # tv norm
    pred_difr = y_pred[:, 2:, 1:-1, :] - y_pred[:, :-2, 1:-1, :]
    pred_difc = y_pred[:, 1:-1, 2:, :] - y_pred[:, 1:-1, :-2, :]
    true_difr = y_true[:, 2:, 1:-1, :] - y_true[:, :-2, 1:-1, :]
    true_difc = y_true[:, 1:-1, 2:, :] - y_true[:, 1:-1, :-2, :]

    return l2_loss(K.sqrt(K.square(pred_difr/2) + K.square(pred_difc/2)+K.epsilon()),
                   K.sqrt(K.square(true_difr/2) + K.square(true_difc/2)+K.epsilon()))

def ssim_loss(masks):
    """Loss function"""
    def ssim(y_true, y_pred):    
        mean = K.constant([103.939, 116.779, 123.68],
                          dtype=tf.float32,
                          shape=[1, 1, 1, 3],
                          name='img_mean')
        c_pred = K.clip(y_pred+mean, 0, 255)
        c_true = K.clip(y_true+mean, 0, 255)
        c_pred = tf.image.rgb_to_grayscale(c_pred)
        c_true = tf.image.rgb_to_grayscale(c_true)
        return K.sum(1-tf.image.ssim(tf.multiply(c_pred, 1-masks), tf.multiply(c_true, 1-masks), 255))
    return ssim

def histogram_loss(y_pred, y_true):
    # histogram loss
    return 0
