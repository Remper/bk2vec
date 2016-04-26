# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import tensorflow as tf
import numpy as np
import os.path

from bk2vec.embeddings import Embeddings
from bk2vec.arguments import EvaluationArguments

args = EvaluationArguments().show_args().args


def restore_evaluation(filename):
    pages = dict()
    categories = 0
    with gzip.open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
        for row in reader:
            if len(row) == 0:
                continue
            index = int(row[0])
            pages[index] = np.array(map(int, row[1:]))
            categories += len(pages[index])
    print("Loaded", len(pages.values()), "pages with", categories, "categories")
    return pages


if not os.path.exists(args.test):
    print("Categories file doesn't exist: " + args.test)
    exit()
if not os.path.exists(args.embeddings):
    print("Embeddings file doesn't exist: " + args.embeddings)
    exit()
print("Restoring test set")
pages = restore_evaluation(args.test)
print("Restoring embeddings")
embeddings, rev_dict = Embeddings.restore(args.embeddings)
print("Restored embeddings with shape:", embeddings.shape)

graph = tf.Graph()


def matrix_distance(tensor1, tensor2):
    with tf.name_scope("matrix_distance"):
        sub = tf.sub(tensor1, tensor2)
        distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(sub), 1), 1e-10, 1e+37))
        return distance

log = open(args.test+'.log', 'wb')
log.write(str(vars(args)))
log.write('\n')
def print_log(*args):
    print(*args)
    log.write(" ".join([str(ele) for ele in args]))
    log.write('\n')

with graph.as_default():
    # Input

    # Look up embeddings for inputs.
    category_tensor = tf.placeholder(tf.float32, shape=[None, embeddings.shape[1]])
    page_tensor = tf.placeholder(tf.float32, shape=[None, embeddings.shape[1]])

    category_loss = tf.reduce_mean(matrix_distance(
        category_tensor,
        page_tensor
    ))

    # Graph
    loss = 0.0
    counts = 0
    print("Starting calculating pages loss")
    indices = list()
    page_indices = list()
    last_indices = list()
    last_page_indices = list()
    for page in pages.keys():
        if len(pages[page]) == 0:
            continue
        counts += 1
        for ele in pages[page]:
            indices.append(ele)
            page_indices.append(page)
        if counts % 100 == 0:
            category_input = np.take(embeddings, indices, axis=0)
            page_input = np.take(embeddings, page_indices, axis=0)
            last_indices = indices
            last_page_indices = page_indices
            indices = list()
            page_indices = list()
            with tf.Session(graph=graph) as session:
                loss += session.run(category_loss, feed_dict={category_tensor: category_input, page_tensor: page_input})
        if counts % 1000000 == 0:
            print_log("Last input shape:", category_input.shape, page_input.shape)
            print_log("Last amount of categories:", len(last_page_indices))
            print_log("Last pages -> categories:")
            for idx in range(20):
                print(rev_dict[last_page_indices[idx]], '->', rev_dict[last_indices[idx]])
            print_log("  " + str(counts // 1000) + "k pages parsed")
            print_log("  Avg loss:", loss / (counts / 100))

    print_log("Average loss: ", loss / (counts / 100))

log.close()