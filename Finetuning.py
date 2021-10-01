import os.path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.optimizers import Nadam
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten
from keras.callbacks import LearningRateScheduler

#gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

distributional = True
urban = False

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

## Data import

train_file = 'data/train_data.h5'
val_file= 'data/validation_data.h5'

path_data = "data/"
checkpoint_path = "/data/lcz42_votes/LCZ_visualization/checkpoint_full.ckpt"
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

val_data = h5py.File(val_file, 'r')
x_val = np.array(val_data.get("x"))

if distributional:
    y_file = 'data/train_label_distributions_data_full.h5'
    y_data = h5py.File(y_file, 'r')
    y_train = np.array((y_data.get("train_label_distributions")))
    y_val_file = 'data/val_label_distributions_data_full.h5'
    y_val_data = h5py.File(y_val_file, 'r')
    y_val = np.array((y_val_data.get("val_label_distributions")))
else:
    y_train = np.array(train_data.get("y"))
    y_val = np.array(val_data.get("y"))

if urban:
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]
    indices_val = np.where(np.where(y_val == np.amax(y_val, 0))[1] + 1 < 11)[0]
    x_val = x_val[indices_val, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]
        y_val = y_val[indices_val]

## Load pretrained model

x_train = x_train[:,:,:,:3]
x_val = x_val[:,:,:,:3]

### Clear Nans (due to zero votes -> investigate later)
indeces_in = np.where(~np.isnan(np.sum(y_train,axis=1)))[0]

x_train = x_train[indeces_in,:,:,:]
y_train = y_train[indeces_in,:]

indeces_in_val = np.where(~np.isnan(np.sum(y_val,axis=1)))[0]

x_val = x_val[indeces_in_val,:,:,:]
y_val = y_val[indeces_in_val,:]

#y_train = y_train[1:5000,:]
#x_train = x_train[1:5000,:,:,:]

trainNumber=y_train.shape[0]
validationNumber=y_val.shape[0]

lrate = 0.0002
lr_sched = step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
batchSize = 32

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 save_freq='epoch')

model = ResNet50(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

head = model.output
head = AveragePooling2D(pool_size = (1,1))(head)
head = Flatten(name="flatten")(head)
head = Dense(256, activation="relu")(head)
#head = Dense(17, activation="softmax")(head)
head = Dense(17)(head)

model = Model(inputs = model.input, outputs = head)

# Define custom loss & accuracy to treat distributional labels
def cat_ce_distributional(y_true, y_pred):
    #return -tf.reduce_sum(y_true * tf.math.multiply_no_nan(tf.math.log(y_pred), tf.cast(tf.math.is_finite(tf.math.log(y_pred)), dtype=tf.float32)))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred))

if distributional:
    model.compile(loss=cat_ce_distributional, optimizer=Nadam())
else:
    model.compile(loss="categorical_crossentropy", optimizer = Nadam(), metrics=['accuracy'])

model.fit(generator(x_train, y_train, batchSize=batchSize, num=trainNumber),
                validation_data= generator(x_val, y_val, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                steps_per_epoch = trainNumber//batchSize,
                epochs=200,
                max_queue_size=100,
                callbacks=[lr_sched, cp_callback])




