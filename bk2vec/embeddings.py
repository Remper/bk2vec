from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import gzip
import numpy as np
import time


class Embeddings:
    def __init__(self, vocabulary_size, embeddings_size):
        self.vocabulary_size = vocabulary_size
        self.size = embeddings_size
        self._tensor = None
        self._norm_tensor = None

    def tensor(self):
        if self._tensor is None:
            self._tensor = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.size], -1.0, 1.0),
                name="embeddings"
            )
        return self._tensor

    def normalized_tensor(self):
        if self._norm_tensor is None:
            with tf.name_scope("embeddings_normalization"):
                norm = tf.sqrt(tf.reduce_sum(tf.square(self.tensor()), 1, keep_dims=True))
                self._norm_tensor = tf.div(self.tensor(), norm)
        return self._norm_tensor

    def dump(self, filename, norm_tensor, dictionary):
        with open(filename, 'wb') as writer:
            counter = 0
            timestamp = time.time()
            for idx, vector in enumerate(norm_tensor):
                counter += 1
                if idx not in dictionary.rev_dict:
                    print("Bullshit happened: ", idx, vector)
                    return
                if counter % 2000000 == 0:
                    print("  ", str(counter // 1000000) + "m writes (" + ("%.5f" % (time.time() - timestamp)) + "s)")
                    timestamp = time.time()
                writer.write(dictionary.rev_dict[idx])
                writer.write('\t')
                writer.write('\t'.join([str(num) for num in vector]))
                writer.write('\n')

    @staticmethod
    def restore(filename):
        final_embeddings = list()
        embeddings = list()
        count = 0
        with gzip.open(filename, 'rb') as reader:
            timestamp = time.time()
            for line in reader:
                row = line.split('\t')
                embeddings.append(np.array(map(float, row[1:])))
                count += 1
                if count % 2000000 == 0:
                    print("  ", str(count // 1000000) + "m words parsed (" + ("%.5f" % (time.time() - timestamp)) + "s)")
                    timestamp = time.time()
                    final_embeddings.append(np.vstack(embeddings))
                    del embeddings
                    embeddings = list()
                    print("  Squashed")
        final_embeddings.append(np.array(embeddings))
        del embeddings
        return np.vstack(final_embeddings)
