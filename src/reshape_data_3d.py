import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from sklearn.cluster import KMeans

import vggish_input

# This script converts spectrogram data to 3D coordinates

# Produce a batch of log mel spectrogram examples.
audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)

data = input_batch[0]

# Get 2D indexes
x, y = np.where(data)

# Put the amplitude data in the Z coordinate
z = np.empty(len(x))
for index, xy in enumerate(zip(x, y)):
    z[index] = data[xy]

print('length before: ', len(z))
z_average = np.average(z)

filter_bias = 0.7
indexes_filtered = np.where(z > z_average + filter_bias)
z_filtered = z[indexes_filtered]
print('length after: ', len(z_filtered))


reduced_data = np.array(
    (x[indexes_filtered], y[indexes_filtered], z[indexes_filtered])).T

kmeans = KMeans(n_clusters=3).fit(reduced_data)
centroids = kmeans.cluster_centers_
unique_labels = np.unique(kmeans.labels_)

fig = plt.figure()
ax = Axes3D(fig)

for unique_label_index, label in enumerate(unique_labels):
    # Get indexes of data with current label
    indexes = []

    for index, number in enumerate(kmeans.labels_):
        if number == label:
            indexes.append(index)

    # Get data with range [0, 1]
    x = reduced_data[indexes, 0]
    y = reduced_data[indexes, 1]
    z = reduced_data[indexes, 2]
   
    # plot sides
    ax.scatter3D(x, y, z)
    # # Plot
    # rectangle = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rectangle)


plt.title('Significant Parts of Log Mel Spectrogram of a Bird Chirp')
ax.view_init(30, -110)
ax.set_xlabel('Frames')
ax.set_ylabel('Frequency Bands')
ax.set_zlabel('Amplitude')

ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=0.8)

plt.show()
