
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
plt.ioff()

import seaborn as sns

## Load Data

preds = h5py.File("predictions_train.h5", 'r')
preds = np.array(preds["predictions_train"])

train_file = "D:/Data/LCZ_Votes/train_data.h5"
train_data = h5py.File(train_file, 'r')
y_train = np.array(train_data.get("y"))

tsne_file = h5py.File("tsne_train.h5", 'r')
tsne_train = np.array((tsne_file.get("tsne_train")))

labels = np.argmax(y_train, axis=1) + 1
labels = labels.reshape(labels.shape[0], -1)

preds = np.append(preds, labels, axis=1)

## Define plot function

def image_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = 17
    palette = np.hstack([np.array([sns.color_palette("Set1")]),
                         np.array([sns.color_palette("Set2")])[:,[0,2,4,6],:],
                         np.array([sns.color_palette("Set3")])[:,[4,10],:],
                         np.array([sns.color_palette("Greys")])[:,[5],:],
                         np.array([sns.color_palette("OrRd")])[:,[5],:]])[:,:num_classes,:]

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

    for i in (np.unique(colors.astype(np.int))):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i+1), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

## Class medians and means

class_medians = np.empty(2048)

for i in range(17):
    class_medians = np.vstack([class_medians, np.median(preds[preds[:, 2048] == (i+1)][:,:2048], axis=0)])

class_medians = class_medians[1:,:]

class_means = np.empty(2048)

for i in range(17):
    class_means = np.vstack([class_means, np.mean(preds[preds[:, 2048] == (i+1)][:,:2048], axis=0)])

class_means = class_means[1:,:]

## Intra-class distances to median as empirical distributions

num_classes = 17

palette = np.hstack([np.array([sns.color_palette("Set1")]),
                     np.array([sns.color_palette("Set2")])])[:,:num_classes,:]

for i in range(17):
    dists_tmp = np.linalg.norm(preds[preds[:, 2048] == (i+1)][:,:2048] - class_medians[i, :], axis=1)
    n, bins, patches = plt.hist(dists_tmp, 50, density=True)
    plt.show()


## Intra-class distances to mean as empirical distributions

for i in range(17):
    dists_tmp = np.linalg.norm(preds[preds[:, 2048] == (i+1)][:,:2048] - class_means[i, :], axis=1)
    n, bins, patches = plt.hist(dists_tmp, 'auto', density=True)
    plt.show()

## Thresholding t-SNE plot w.r.t. distance to median

dists_all = np.linalg.norm(preds[:,:2048] - class_medians[preds[:, 2048].astype(int) - 1], axis=1)
y_train = np.argmax(y_train, axis=1)

threshold = 2

indeces = np.where(dists_all < threshold)[0]

image_scatter(tsne_train[indeces], y_train[indeces])
plt.show()







