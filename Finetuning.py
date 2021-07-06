import os.path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten
from keras.callbacks import LearningRateScheduler

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def generator(features, labels, batchSize=32, num=None, mode="all"):
    indices=np.arange(num)
    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batchSize):
            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()
            if mode == "urban":
                by = labels[batch_indices, :10]
            else:
                by = labels[batch_indices, :]
            bx = features[batch_indices,:,:,:]
            yield (bx,by)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        print(initial_lr * (decay_factor ** np.floor(epoch/step_size)))
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)

distributional = False
urban = False

## Data import

train_file = 'data/train_data.h5'
path_data = "data/"
checkpoint_path = "/data/lcz42_votes/LCZ_visualization/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

'''
path_finetuning_data = "D:/Data/EuroSat/"

train_fine, valid_fine = tfds.load('eurosat/rgb',
                               split=['train[:70%]', 'train[70%:]'],
                               data_dir=path_finetuning_data,
                               as_supervised=True)
'''

train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("x"))
y_train = np.array(train_data.get("y"))

if urban:
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]

## Load pretrained model

x_train = x_train[:,:,:,:3]

trainNumber=y_train.shape[0]

lrate = 0.002
lr_sched = step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
batchSize = 32

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = ResNet50(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

head = model.output
head = AveragePooling2D(pool_size = (1,1))(head)
head = Flatten(name="flatten")(head)
head = Dense(256, activation="relu")(head)
head = Dense(17, activation="softmax")(head)

model = Model(inputs = model.input, outputs = head)

model.compile(loss="categorical_crossentropy", optimizer = Nadam(), metrics=['accuracy'])

model.fit(generator(x_train, y_train, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                epochs=300,
                max_queue_size=100,
                callbacks=[lr_sched, cp_callback])

