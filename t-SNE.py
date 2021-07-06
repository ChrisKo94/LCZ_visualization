
import h5py
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

urban = False

## Data import

train_file = '/data/lcz42_votes/train_data.h5'
path_data = "/data/lcz42_votes/"

train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("x"))
y_train = np.array(train_data.get("y"))

if urban:
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]

## Load pretrained model

model = ResNet50(include_top=False, weights='custom', input_shape=(32, 32, 3))

preds = model.predict(x_train).reshape((x_train.shape[0],-1))

## t-SNE

def image_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = 17  # len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

image_tsne = TSNE(random_state=RS).fit_transform(preds)

image_scatter(image_tsne, y_train)
plt.show()
