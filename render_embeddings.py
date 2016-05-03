# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os

from bk2vec.embeddings import Embeddings
from bk2vec.arguments import RenderArguments

args = RenderArguments().show_args().args

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
    exit()

print("Restoring embeddings")
embeddings, rev_dict, _ = Embeddings.restore(args.embeddings)
print("Restored embeddings with shape:", embeddings.shape)


def render(filename):
    print("Rendering", filename)
    indices = list()
    with open(filename, 'rb') as file:
        for line in file.readlines():
            indices.append(int(line.split('\t')[0]))

    vectors = list()
    labels = list()
    for index in indices:
        vectors.append(embeddings[index])
        labels.append(repr(rev_dict[index]))

    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        plt.figure(figsize=(18, 18))  #in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i,:]
            plt.scatter(x, y)
            plt.grid(True)
            plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        plt.ylim([-1000, 1000])
        plt.xlim([-1000, 1000])

        plt.savefig(filename)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    filtered = np.nan_to_num(vectors)
    print("Before transform: ", filtered.shape)
    low_dim_embs = tsne.fit_transform(filtered)
    print("After transform: ", low_dim_embs.shape, len(labels))
    plot_with_labels(low_dim_embs, labels, filename+'.png')

if args.indices.endswith('/'):
    for path in os.listdir(args.indices):
        if path.endswith('.txt'):
            render(args.indices+path)
else:
    render(args.indices)