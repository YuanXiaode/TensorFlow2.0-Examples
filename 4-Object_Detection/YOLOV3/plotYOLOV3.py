## 画出网络结构，显示网络层和变量

import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

input_size   = 416

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)
model = tf.keras.Model(input_layer, feature_maps)
# tf.keras.utils.plot_model(model,"name.jpg",show_shapes = True)

for l in model.layers:
    print("=====",l.name.ljust(50))
    for v in l.trainable_variables:
        print(" "*30,v.name.ljust(50), v.shape)

# for l in model.trainable_variables:
#     print(l.name.ljust(50),l.shape)