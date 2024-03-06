# -*- coding: utf-8 -*-
# -------------------------------------------------------
# メモリの制限 tensorflow-gpu (2.0.0)
# -------------------------------------------------------
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

# -------------------------------------------------------

import numpy as np
import os
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers

import pickle
import matplotlib.pyplot as plt

vgg16 =VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))

import random
classes = ['dog', 'cat']

def build_transfer_model(vgg16):
  model = Sequential(vgg16.layers)
  for layer in model.layers[:15]:
    layer.trainable = False
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(len(classes), activation='softmax'))
  return model

model = vgg16
model.summary()
#model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.SGD(learning_rate=1e-3, momentum=0.9),
              metrics=['accuracy'])

idg_train = ImageDataGenerator(rescale=1/255.,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               preprocessing_function=preprocess_input
)

idg_valid = ImageDataGenerator(rescale=1/255.)

img_itr_train = idg_train.flow_from_directory(directory='./animals/train/',
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              classes=classes,
                                              batch_size=1,
                                              class_mode='categorical',
                                              shuffle=True
)

img_itr_validation = idg_valid.flow_from_directory(directory='./animals/test/',
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   classes=classes,
                                                   batch_size=1,
                                                   class_mode='categorical',
                                                   shuffle=True
)


from datetime import datetime

model_dir = os.path.join('./animals/',
                         datetime.now().strftime('%y%m%d_%H%M')
)

os.makedirs(model_dir, exist_ok = True)
print('model_dir:', model_dir)

dir_weights = os.path.join(model_dir, 'weights')
os.makedirs(dir_weights, exist_ok = True)

from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint
import math
file_name='vgg16_fine'
batch_size_train=16
batch_size_validation=16

steps_per_epoch=math.ceil(img_itr_train.samples/batch_size_train
)
validation_steps=math.ceil(img_itr_validation.samples/batch_size_validation
)

cp_filepath = os.path.join(dir_weights, 'ep_{epoch:02d}_ls_{loss:.1f}.h5')

cp = ModelCheckpoint(cp_filepath,
                     monitor='loss',
                     verbose=0,
                     save_best_only=False,
                     save_weights_only=True,
                     mode='auto',
                     save_freq=5
)
csv_filepath = os.path.join(model_dir, 'loss.csv')
csv = CSVLogger(csv_filepath, append=True)

hist=model.fit(
    img_itr_train,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=1,
    validation_data=img_itr_validation,
    validation_steps=validation_steps,
    shuffle=True,
    callbacks=[cp, csv]
)

model.save(file_name+'.h5')
