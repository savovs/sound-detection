import os
import numpy as np
from sklearn.cluster import KMeans

import params

def calculate_boxes(spectrogram):
    """ Returns a list of box coordinates like this: [(x_min, x_max, y_min, y_max), ...]"""

    # Get 2D indexes
    x, y = np.where(spectrogram)

    # Put the amplitude spectrogram in the Z coordinate
    z = np.empty(len(x))
    for index, xy in enumerate(zip(x, y)):
        z[index] = spectrogram[xy]

    z_mean = np.mean(z)

    # Remove points where amplitude is lower than the mean + bias
    bias = abs(z_mean) * params.AMPLITUDE_FILTER_BIAS_MAGNITUDE
    indexes_filtered = np.where(z > z_mean + bias)

    point_threshold = 200

    # Reduce the filter bias until there is enough points to satisfy KMeans
    while indexes_filtered[0].shape[0] < point_threshold:
        bias -= 0.2
        point_threshold -= 70
        indexes_filtered = np.where(z > z_mean + bias)

        if point_threshold < 10:
            return

    z_filtered = z[indexes_filtered]

    reduced_data = np.array((x[indexes_filtered], y[indexes_filtered], z[indexes_filtered])).T

    try:
        kmeans = KMeans(n_clusters=params.N_CLUSTERS).fit(reduced_data)
        centroids = kmeans.cluster_centers_
        unique_labels = np.unique(kmeans.labels_)

        boxes = []
        for unique_label_index, label in enumerate(unique_labels):
            # Get indexes of spectrogram with current label
            indexes = []

            for index, number in enumerate(kmeans.labels_):
                if number == label:
                    indexes.append(index)

            x = reduced_data[indexes, 0]
            y = reduced_data[indexes, 1]

            box = (min(x), max(x), min(y), max(y))
            box = list(map(int, box))
            boxes.append(box)

        return boxes

    except Exception as e:
        print('Whoops, error:\n', str(e), '\nBoxes won\'t be returned.')
        pass
    


