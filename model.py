import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.keras import backend, models, layers
from tensorflow.python.keras.applications import imagenet_utils

from utils import custom_sample_wights, custom_nme

class CustomNME(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None):
        super(CustomNME, self).__init__(name, dtype)
        self.per_image_nme = []

    def get_config(self):
        return super(CustomNME, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def update_state(self, y_true, y_pred, sample_weight=None):
        l2_dist = tf.reduce_mean(tf.math.sqrt((y_true - y_pred) * (y_true - y_pred)), axis=1)
        norm_coeff = tf.map_fn(
            fn = lambda y : tf.math.sqrt((y[90] - y[72]) * (y[90] - y[72]) +
                                         (y[91] - y[73]) * (y[91] - y[73])),
            elems = y_true)
        self.per_image_nme.append(tf.reduce_mean(l2_dist / norm_coeff))

    def result(self):
        return np.mean(self.per_image_nme)
    
    def reset_state(self):
        self.per_image_nme = []

class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def get_config(self):
        return super(CustomModel, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def train_step(self, data):
        x, y = data
        y_landmarks, y_angles = y

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            y_pred_array = y_pred.numpy()
            y_angles_array = y_angles.numpy()
            weights = [ 100 * custom_sample_wights(euler_angle_gt, landmarks)
                            for euler_angle_gt, landmarks in zip(y_angles_array, y_pred_array)
            ]
            loss = self.mse_loss(y_landmarks, y_pred, sample_weight = weights)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y_landmarks, y_pred)
        metrics = {'loss': np.mean(loss)}
        metrics.update({ m.name: m.result() for m in self.metrics })

        return metrics 

    def test_step(self, data):
        x, y = data
        y_landmarks, y_angles = y
        y_pred = self(x, training=False)

        y_pred_array = y_pred.numpy()
        y_angles_array = y_angles.numpy()
        weights = [ custom_sample_wights(euler_angle_gt, landmarks)
                        for euler_angle_gt, landmarks in zip(y_angles_array, y_pred_array)
        ]
        loss = self.mse_loss(y_landmarks, y_pred, sample_weight = weights)
        
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {'loss': np.mean(loss)}
        metrics.update({ m.name: m.result() for m in self.metrics })

        return metrics 

def create_landmarks_detector(input_shape = None):
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

    img_input = layers.Input(shape=input_shape)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(img_input)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = depth(backend.int_shape(x)[channel_axis] * 6)

    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    features_1 = layers.Flatten()(layers.GlobalAveragePooling2D()(x))
    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding='same',
        use_bias=False,
        name='Conv_2')(x)
    features_2 = layers.Flatten()(layers.GlobalAveragePooling2D()(x))
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)
    x = layers.Conv2D(
        128,
        kernel_size=7,
        padding='same',
        use_bias=True,
        name='Conv_3')(x)
    x = activation(x)
    features_3 = layers.Flatten()(x)
    concat = layers.concatenate([features_1, features_2, features_3], axis=1)
    output = layers.Dense(2*68)(concat)

    model = CustomModel(img_input, output, name='LandmarksDetector')
    
    return model


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])

def depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
            depth(filters * se_ratio),
            kernel_size=1,
            padding='same',
            name=prefix + 'squeeze_excite/Conv')(
                    x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            name=prefix + 'squeeze_excite/Conv_1')(
                    x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x

def stack_fn(x, kernel, activation, se_ratio):
    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
    x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
    return x

def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                                                activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
                depth(infilters * expansion),
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=prefix + 'expand')(
                        x)
        x = layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand/BatchNorm')(
                        x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
                padding=imagenet_utils.correct_pad(x, kernel_size),
                name=prefix + 'depthwise/pad')(
                        x)
    x = layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding='same' if stride == 1 else 'valid',
            use_bias=False,
            name=prefix + 'depthwise')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'depthwise/BatchNorm')(
                    x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'project')(
                    x)
    x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'project/BatchNorm')(
                    x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x
