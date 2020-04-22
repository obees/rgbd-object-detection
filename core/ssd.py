import tensorflow as tf
from core.models.resnet import ResNet50
from .models.mobilenet_v3 import *
from configuration import NUM_CLASSES, ASPECT_RATIOS


class SSD(tf.keras.Model):
    def __init__(self):
        super(SSD, self).__init__()
        self.num_classes = NUM_CLASSES + 1
        self.anchor_ratios = ASPECT_RATIOS

        # ResNet50 version
        # self.backbone = ResNet50()
        
        # MobilenetV2 version
        #self.mobilenetv2_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        #self.mobilenetv2_model.trainable = False
        #self.branch1_backbone = self.mobilenetv2_model.get_layer("block_13_expand_relu").output # (14,14,  576)
        #self.x_backbone = self.mobilenetv2_model.output                                         # ( 7, 7, 1280)
        
        # MobilenetV3 version
        self.mobilenetv3_model = MobileNetV3Small(backend=tf.keras.backend,layers=tf.keras.layers,models=tf.keras.models,utils=tf.keras.utils,input_shape=(224,224,3), include_top=False, weights='imagenet')
        self.mobilenetv3_model.trainable = False
        #for layer in self.mobilenetv3_model.layers:
        #    print(layer.name)
        self.branch1_backbone = self.mobilenetv3_model.get_layer("expanded_conv_8/activation1").output # (14,14,  576)
        self.x_backbone = self.mobilenetv3_model.output                                  # ( 7, 7, 576)
        
        self.features_list = [self.branch1_backbone, self.x_backbone]
        # self.features_extraction_model = keras.Model(inputs=mobilenetv2_model.input, outputs=features_list)
        #self.backbone = tf.keras.Model(inputs=self.mobilenetv2_model.input, outputs=self.features_list)
        self.backbone = tf.keras.Model(inputs=self.mobilenetv3_model.input, outputs=self.features_list)
        self.backbone.trainable = False
        
        #self.conv1 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same", name="conv1")
        #self.conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same", name="conv2_1")
        #self.conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same", name="conv2_2")
        #self.conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", name="conv3_1")
        #self.conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same", name="conv3_2")
        #self.conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", name="conv4_1")
        #self.conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same", name="conv4_2")
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name="pool")

        self.conv1s = tf.keras.layers.SeparableConv2D(filters=576, kernel_size=(1, 1), strides=1, padding="same", name="conv1")
        self.conv2_1s = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(1, 1), strides=1, padding="same", name="conv2_1")
        self.conv2_2s = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same", name="conv2_2")
        self.conv3_1s = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", name="conv3_1")
        self.conv3_2s = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same", name="conv3_2")
        #self.conv4_1s = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same", name="conv4_1")
        #self.conv4_2s = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same", name="conv4_2")
        
        self.predict_1 = self._predict_layer(k=self._get_k(i=0),j=0)
        self.predict_2 = self._predict_layer(k=self._get_k(i=1),j=1)
        self.predict_3 = self._predict_layer(k=self._get_k(i=2),j=2)
        self.predict_4 = self._predict_layer(k=self._get_k(i=3),j=3)
        #self.predict_5 = self._predict_layer(k=self._get_k(i=4),j=4)
        self.predict_6 = self._predict_layer(k=self._get_k(i=5),j=5)

    def _predict_layer(self, k,j):
        filter_num = k * (self.num_classes + 4)
        return tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding="same", name="predict"+str(j))

    def _get_k(self, i):
        # k is the number of boxes generated at each position of the feature map.
        return len(self.anchor_ratios[i]) + 1

    def call(self, inputs, training=None, mask=None):
        branch_1, x = self.backbone(inputs, training=False)
        predict_1 = self.predict_1(branch_1) # (14, 14, 100)

        x = self.conv1s(x) # (7, 7, 1280)
        branch_2 = x
        predict_2 = self.predict_2(branch_2) # (7,7,150)
        

        x = self.conv2_1s(x)
        x = self.conv2_2s(x)
        branch_3 = x
        predict_3 = self.predict_3(branch_3) # (4, 4, 150)
        

        x = self.conv3_1s(x)
        x = self.conv3_2s(x)
        branch_4 = x
        predict_4 = self.predict_4(branch_4)
        #print(predict_4.shape)

        #x = self.conv4_1s(x)
        #x = self.conv4_2s(x)
        #branch_5 = x
        #predict_5 = self.predict_5(branch_5)
        #print(predict_5.shape)

        branch_6 = self.pool(x)
        branch_6 = tf.expand_dims(input=branch_6, axis=1)
        branch_6 = tf.expand_dims(input=branch_6, axis=2)
        predict_6 = self.predict_6(branch_6)
        #print(predict_6.shape)

        # predict_i shape : (batch_size, h, w, k * (c+4)), where c is self.num_classes.
        #return [predict_1, predict_2, predict_3, predict_4, predict_5, predict_6]
        return [predict_1, predict_2, predict_3, predict_4, predict_6]


def ssd_prediction(feature_maps, num_classes):
    batch_size = feature_maps[0].shape[0]
    predicted_features_list = []
    for feature in feature_maps:
        predicted_features_list.append(tf.reshape(tensor=feature, shape=(batch_size, -1, num_classes + 4)))
    predicted_features = tf.concat(values=predicted_features_list, axis=1)
    return predicted_features
