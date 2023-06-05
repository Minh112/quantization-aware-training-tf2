from keras.layers import *
from keras.models import Model
import tensorflow as tf


class MobileNet:
    def __init__(self, size=224):
        self.size = size

    def conv_bn_relu(self, filters, kernel_size, strides, padding, x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def dwconv_bn_relu(self, kernel_size, strides, padding, x):
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def depthwise_separable_convolution(self, filters, strides, x):
        x = self.dwconv_bn_relu(kernel_size=(3, 3), strides=strides, padding='same', x=x)
        x = self.conv_bn_relu(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', x=x)
        return x

    def __call__(self):
        inputs = Input(shape=(self.size, self.size, 3))
        x = self.conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', x=inputs)
        x = self.depthwise_separable_convolution(filters=64, strides=(1, 1), x=x)
        x = self.depthwise_separable_convolution(filters=128, strides=(2, 2), x=x)
        x = self.depthwise_separable_convolution(filters=128, strides=(1, 1), x=x)
        x = self.depthwise_separable_convolution(filters=256, strides=(2, 2), x=x)
        x = self.depthwise_separable_convolution(filters=256, strides=(1, 1), x=x)
        x = self.depthwise_separable_convolution(filters=512, strides=(2, 2), x=x)
        for i in range(5):
            x = self.depthwise_separable_convolution(filters=512, strides=(1, 1), x=x)
        x = self.depthwise_separable_convolution(filters=1024, strides=(2, 2), x=x)
        x = self.depthwise_separable_convolution(filters=1024, strides=(1, 1), x=x)
        x = AveragePooling2D(pool_size=(self.size//32, self.size//32))(x)
        x = Dense(1000, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        return model



class MobileNetV2:
    def __init__(self, size=224):
        self.size = size

    def conv_bn_relu(self, filters, kernel_size, strides, padding, x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(6)(x)
        return x

    def dwconv_bn_relu(self, kernel_size, strides, padding, x):
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(6)(x)
        return x

    def bottleneck(self, t, in_c, out_c, strides, x):
        x1 = x
        x = self.conv_bn_relu(filters=t*in_c, kernel_size=(1, 1), strides=(1, 1), padding='valid', x=x1)
        x = self.dwconv_bn_relu(kernel_size=(3, 3), strides=strides, padding='same', x=x)
        x = Conv2D(filters=out_c, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
        x = BatchNormalization()(x)
        if in_c == out_c and strides == (1, 1):
            x = Add()([x1, x])
        return x

    def sequence_bottleneck(self, t, in_c, out_c, strides, n, x):
        x = self.bottleneck(t, in_c, out_c, strides, x)
        for i in range(n-1):
            x = self.bottleneck(t, out_c, out_c, (1, 1), x)
        return x

    def __call__(self):
        inputs = Input(shape=(self.size, self.size, 3))
        x = self.conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', x=inputs)
        x = self.dwconv_bn_relu(kernel_size=(3, 3), strides=(1, 1), padding='same', x=x)
        x = self.conv_bn_relu(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', x=x)
        x = self.sequence_bottleneck(t=6, in_c=16, out_c=24, strides=(2, 2), n=2, x=x)
        x = self.sequence_bottleneck(t=6, in_c=24, out_c=32, strides=(2, 2), n=3, x=x)
        x = self.sequence_bottleneck(t=6, in_c=32, out_c=64, strides=(2, 2), n=4, x=x)
        x = self.sequence_bottleneck(t=6, in_c=64, out_c=96, strides=(1, 1), n=3, x=x)
        x = self.sequence_bottleneck(t=6, in_c=96, out_c=160, strides=(2, 2), n=3, x=x)
        x = self.sequence_bottleneck(t=6, in_c=160, out_c=320, strides=(1, 1), n=1, x=x)
        x = self.conv_bn_relu(filters=1280, kernel_size=(1, 1), strides=(1, 1), padding='valid', x=x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1000, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=x)
        return model



class MobileNetV3:
    def __init__(self, size=224, type='large'):
        self.size = size
        self.type = type

    def h_swish(self, x):
        x1 = tf.add(x, 3)
        x1 = ReLU(6)(x1)
        x1 = tf.multiply(x1, 1/6)
        x = Multiply()([x, x1])
        return x

    def conv_bn_relu(self, filters, kernel_size, strides, padding, NL_used, x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        if NL_used == "RE":
            x = ReLU()(x)
        elif NL_used == "HS":
            x = self.h_swish(x)
        return x

    def dwconv_bn_relu(self, kernel_size, strides, padding, NL_used, x):
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        if NL_used == "RE":
            x = ReLU()(x)
        elif NL_used == "HS":
            x = self.h_swish(x)
        return x

    def SE_branch(self, expand_c, SE_c, x):
        x1 = GlobalAveragePooling2D(keepdims=True)(x)
        x1 = Conv2D(filters=SE_c, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(x1)
        x1 = ReLU()(x1)
        x1 = Conv2D(filters=expand_c, kernel_size=(1, 1), strides=(1, 1), use_bias=True)(x1)
        x1 = tf.add(x1, 3)
        x1 = ReLU(6)(x1)
        x1 = tf.multiply(x1, 1/6)
        x = Multiply()([x, x1])
        return x


    def bottleneck(self, kernel_size, in_c, expand_c, out_c, SE_used, NL_used, strides, x):
        x1 = self.conv_bn_relu(filters=expand_c, kernel_size=(1, 1), strides=(1, 1), padding='valid', NL_used=NL_used,
                               x=x)
        x1 = self.dwconv_bn_relu(kernel_size=kernel_size, strides=strides, padding='same', NL_used=NL_used, x=x1)
        if SE_used:
            x1 = self.SE_branch(expand_c, SE_used, x1)
        x1 = self.conv_bn_relu(filters=out_c, kernel_size=(1, 1), strides=(1, 1), padding='valid', NL_used='none', x=x1)
        if in_c == out_c and strides == (1, 1):
            x = Add()([x, x1])
        else:
            x = x1
        return x


    def first_bottleneck(self, x):
        x1 = self.dwconv_bn_relu(kernel_size=(3, 3), strides=(1, 1), padding='same', NL_used='RE', x=x)
        x1 = self.conv_bn_relu(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', NL_used='none', x=x1)
        x = Add()([x, x1])
        return x


    def __call__(self):
        inputs = Input(shape=(self.size, self.size, 3))
        x = self.conv_bn_relu(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', NL_used='HS', x=inputs)
        x = self.first_bottleneck(x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=16, expand_c=64, out_c=24, SE_used=False, NL_used='RE',
                            strides=(2, 2), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=24, expand_c=72, out_c=24, SE_used=False, NL_used='RE',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=24, expand_c=72, out_c=40, SE_used=24, NL_used='RE',
                            strides=(2, 2), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=40, expand_c=120, out_c=40, SE_used=32, NL_used='RE',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=40, expand_c=120, out_c=40, SE_used=32, NL_used='RE',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=40, expand_c=240, out_c=80, SE_used=False, NL_used='HS',
                            strides=(2, 2), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=80, expand_c=200, out_c=80, SE_used=False, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=80, expand_c=184, out_c=80, SE_used=False, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=80, expand_c=184, out_c=80, SE_used=False, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=80, expand_c=480, out_c=112, SE_used=120, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(3, 3), in_c=112, expand_c=672, out_c=112, SE_used=168, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=112, expand_c=672, out_c=160, SE_used=168, NL_used='HS',
                            strides=(2, 2), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=160, expand_c=960, out_c=160, SE_used=240, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.bottleneck(kernel_size=(5, 5), in_c=160, expand_c=960, out_c=160, SE_used=240, NL_used='HS',
                            strides=(1, 1), x=x)
        x = self.conv_bn_relu(filters=960, kernel_size=(1, 1), strides=(1, 1), padding='valid', NL_used='HS', x=x)
        x = GlobalAveragePooling2D(keepdims=True)(x)
        x = Conv2D(filters=1280, kernel_size=1, strides=1, padding='valid', use_bias=True)(x)
        x = self.h_swish(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=100, kernel_size=1, strides=1, padding='valid', use_bias=True)(x)
        x = Flatten()(x)
        x = Softmax()(x)
        model = Model(inputs=inputs, outputs=x)
        return model



if __name__ == "__main__":
    model = MobileNetV3(224)()
    model.summary()
    model.save("saved_models/mobilenetv3-first.h5")
