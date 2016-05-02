# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

from bk2vec.embeddings import Embeddings
from bk2vec.arguments import EvaluationArguments

args = EvaluationArguments().show_args().args

print("Restoring embeddings")
embeddings, rev_dict, dictionary = Embeddings.restore(args.embeddings)
print("Restored embeddings with shape:", embeddings.shape)

distances = list()
labels = list()
words1 = list()
words2 = list()
with open(args.test, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, quotechar='')
    count = 0
    for row in reader:
        count += 1
        if count == 1:
            print("Skipping header")
            continue
        if len(row) != 3:
            print("Inconsistent file")
            continue
        if row[0] not in dictionary:
            print("Word", row[0], "not in dictionary")
            continue
        if row[1] not in dictionary:
            print("Word", row[1], "not in dictionary")
            continue
        word1 = embeddings[dictionary[row[0]]]
        word2 = embeddings[dictionary[row[1]]]
        words1.append(row[0])
        words2.append(row[1])
        distance = np.linalg.norm(word1-word2)
        distances.append(distance)
        labels.append(10.0-float(row[2]))
with open(args.test+"_distances", "wb") as file:
    for distance in distances:
        file.write(str(distance))
        file.write('\n')
with open(args.test+"_labels", "wb") as file:
    for label in labels:
        file.write(str(label))
        file.write('\n')
distances = normalize(distances).reshape(-1, 1)
labels = normalize(labels).reshape(-1, 1)
with open(args.test+"_distances_normalized", "wb") as file:
    for distance in distances:
        file.write(str(distance))
        file.write('\n')
with open(args.test+"_labels_normalized", "wb") as file:
    for label in labels:
        file.write(str(label))
        file.write('\n')
print(distances.shape, labels.shape)
for idx in range(10):
    print(words1[idx], words2[idx], distances[idx], labels[idx])
print("Pearson:", pearsonr(distances, labels))
print("Spearman:", spearmanr(distances, labels))
