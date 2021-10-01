
import h5py
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

urban = False
distributional = True

## Data import

train_file = '/data/lcz42_votes/data/train_data.h5'
train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("x"))

test_file = '/data/lcz42_votes/data/test_data.h5'
test_data = h5py.File(test_file, 'r')
x_test = np.array(test_data.get("x"))

path_data = "/data/lcz42_votes/"

if distributional:
    checkpoint_path = "/data/lcz42_votes/model_checkpoints/checkpoint_full.ckpt"
else:
    checkpoint_path = "/data/lcz42_votes/model_checkpoints/checkpoint.ckpt"

if distributional:
    y_train_file = '/data/lcz42_votes/data/train_label_distributions_data_full.h5'
    y_data = h5py.File(y_train_file, 'r')
    y_train = np.array((y_data.get("train_label_distributions")))
    y_test_file = '/data/lcz42_votes/data/test_label_distributions_data_full.h5'
    y_data_test = h5py.File(y_test_file, 'r')
    y_test = np.array((y_data_test.get("test_label_distributions")))
else:
    y_train = np.array(train_data.get("y"))
    y_test = np.array((test_data.get("y")))

if urban:
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]
    indices_test = np.where(np.where(y_test == np.amax(y_test, 0))[1] + 1 < 11)[0]
    x_test = x_test[indices_test, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]
        y_test = y_test[indices_test]

## Load pretrained model

model = ResNet50(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

#head = model.output
#head = AveragePooling2D(pool_size = (1,1))(head)
#head = Flatten(name="flatten")(head)
#head = Dense(256, activation="relu")(head)
#head = Dense(17, activation="softmax")(head)
#model = Model(inputs = model.input, outputs = head)

model.load_weights(checkpoint_path)

### Clear Nans (due to zero votes -> investigate later)
indeces_in = np.where(~np.isnan(np.sum(y_train,axis=1)))[0]
indeces_in_test = np.where(~np.isnan(np.sum(y_test,axis=1)))[0]

x_train = x_train[indeces_in,:,:,:3]
y_train = y_train[indeces_in,:]
x_test = x_test[indeces_in_test,:,:,:3]
y_test = y_test[indeces_in_test,:]

preds = model.predict(x_train).reshape((x_train.shape[0],-1))
preds_test = model.predict(x_test).reshape((x_test.shape[0],-1))

if distributional:
    preds_h5 = h5py.File("/data/lcz42_votes/LCZ_visualization/predictions_train_full.h5", 'w')
    preds_h5.create_dataset('predictions_train', data=preds)
    preds_h5.close()
    preds_test_h5 = h5py.File("/data/lcz42_votes/LCZ_visualization/predictions_test_full.h5", 'w')
    preds_test_h5.create_dataset('predictions_test', data=preds_test)
    preds_test_h5.close()

#if distributional:
#    preds = h5py.File("/data/lcz42_votes/LCZ_visualization/predictions_train_full.h5", 'r')
#    preds = np.array(preds["predictions_train"])
#else:
#    preds = h5py.File("/data/lcz42_votes/LCZ_visualization/predictions_train.h5", 'r')
#    preds = np.array(preds["predictions_train"])

## t-SNE

def image_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = 17  # len(np.unique(colors))
    palette = np.hstack([np.array([sns.color_palette("Set1")]),
                         np.array([sns.color_palette("Set2")])])[:,:num_classes,:]

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[0,colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax.legend(handles = [mpatches.Patch(color = palette[0,i], label = str(i+1)) for i in range(17)],
              ncol = 2)

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i+1), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

if distributional:
    image_tsne = TSNE(random_state=RS).fit_transform(preds)
    image_tsne_full_h5 = h5py.File("tsne_train_full.h5", 'w')
    image_tsne_full_h5.create_dataset("tsne_train_full", data=image_tsne)
    image_tsne_full_h5.close()
    image_tsne_test = TSNE(random_state=RS).fit_transform(preds_test)
    image_tsne_test_full_h5 = h5py.File("tsne_test_full.h5", 'w')
    image_tsne_test_full_h5.create_dataset("tsne_test_full", data=image_tsne_test)
    image_tsne_test_full_h5.close()

else:
    image_tsne = TSNE(random_state=RS).fit_transform(preds)
    image_tsne_h5 = h5py.File("tsne_train.h5", 'w')
    image_tsne_h5.create_dataset("tsne_train", data=image_tsne)
    image_tsne_h5.close()
    image_tsne_test = TSNE(random_state=RS).fit_transform(preds_test)
    image_tsne_test_h5 = h5py.File("tsne_test.h5", 'w')
    image_tsne_test_h5.create_dataset("tsne_test", data=image_tsne_test)
    image_tsne_test_h5.close()

#image_tsne_h5 = h5py.File("tsne_train.h5", 'r')
#image_tsne = np.array(image_tsne_h5.get("tsne_train"))

# Import labels (NOT distributions) if needed

if distributional:
    y_train = np.array(train_data.get("y"))
    y_test = np.array((test_data.get("y")))
    # Subset w.r.t. earlier derived index list to get matching tables
    y_train = y_train[indeces_in, :]
    y_test = y_test[indeces_in_test, :]
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

image_scatter(image_tsne, y_train)
plt.show()
