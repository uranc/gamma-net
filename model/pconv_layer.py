from keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Conv2D, Multiply
from tensorflow.keras.initializers import Constant
import numpy as np
import tensorflow as tf

class PConv2D(Conv2D):
    """2D convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Arguments:
  """
    def __init__(self, filters, kernel_size, output_dim=2, **kwargs):
        self.output_dim = output_dim
        super().__init__(filters, kernel_size, **kwargs)
        self.input_spec = [
            InputSpec(ndim=self.rank+2), InputSpec(ndim=self.rank+2)]

    # def get_config(self):
    #     return super().get_config()

    def build(self, input_shape):
        assert isinstance(input_shape, list)  # assert multi inputs
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_shape = [input_s.as_list() for input_s in input_shape]
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        self.input_dim = input_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        else:
            self.bias = None
        # self.mask_kernel = self.kernel
        self.mask_kernel = K.constant(np.ones(kernel_shape))
        # print(kernel_shape)
        # self.mask_kernel = self.add_weight(
        #         name='ones',
        #         shape=kernel_shape,
        #         initializer=Constant(np.ones(kernel_shape)),
        #         trainable=False)
        # self.mask_kernel(trainable=True)
        # print(self.mask_kernel)
        # print(kernel_shape)
        # self.mask_kernel = tf.Variable(np.ones(kernel_shape), trainable=False)
        self.built = True

    def call(self, inputs):
        assert isinstance(inputs, list)  # assert multi inputs
        # inp_masked = Multiply()([inputs[0], inputs[1]])
        outputs = K.conv2d(inputs[0]*inputs[1],
                           self.kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate
                           )

        mask_outputs = K.conv2d(inputs[1],
                                self.mask_kernel,
                                strides=self.strides,
                                padding=self.padding,
                                data_format=self.data_format,
                                dilation_rate=self.dilation_rate
                                )

        # renormalize weights
        norm_vgg = self.kernel_size[0] * \
            self.kernel_size[1] * self.kernel_size[1]
        # norm_vgg = 1.
        mask_weights = (norm_vgg/(mask_outputs + K.epsilon()))
        mask_out = K.cast(K.greater(mask_outputs, 0), 'float32')
        outputs *= (mask_weights*mask_out)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return [outputs, mask_out]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0], ) + \
                tuple(new_space) + (self.filters, )
            return [new_shape, new_shape]
        else:
            space = input_shape[0][2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
                new_shape = (input_shape[0][0], )(
                    self.filters, ) + tuple(new_space)
            return [new_shape, new_shape]
