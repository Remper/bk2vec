# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import tensorflow as tf
import numpy as np
import os.path

#CATEGORIES = '/Users/remper/Projects/bk2vec/borean/embeddings_80_500k_traintest_test.gz'
#EMBEDDINGS = '/Users/remper/Projects/bk2vec/borean/embeddings_80_500k_traintest.tsv.gz'
# Borean
#CATEGORIES = '/borean1/data/pokedem-experiments/bk2vec/embeddings_80_500k_traintest_test.gz'
#EMBEDDINGS = '/borean1/data/pokedem-experiments/bk2vec/embeddings_80_500k_traintest.tsv.gz'
# Auster
CATEGORIES = 'embeddings_80_500k_traintest_test.gz'
EMBEDDINGS = 'embeddings_80_500k_traintest.tsv.gz'

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


def restore_embeddings(filename):
  embeddings = list()
  final_embeddings = list()
  embeddings_size = 0
  embeddings_count = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    try:
      for row in reader:
        if embeddings_size == 0:
          embeddings_size = len(row) - 1
        embeddings_count += 1
        embedding = np.array(map(float, row[1:]))
        embeddings.append(embedding)
        if embeddings_count % 2000000 == 0:
          print("  "+str(embeddings_count // 1000000)+"m words parsed")
          final_embeddings.append(np.array(embeddings))
          del embeddings
          embeddings = list()
          print("  Squashed")
      final_embeddings.append(np.array(embeddings))
      final_embeddings = np.vstack(final_embeddings)
      del embeddings
    except csv.Error:
      print(u"Dunno why this error happens")
  return final_embeddings, embeddings_size

if not os.path.exists(CATEGORIES):
  print("Categories file doesn't exist: "+CATEGORIES)
  exit()
if not os.path.exists(EMBEDDINGS):
  print("Embeddings file doesn't exist: "+EMBEDDINGS)
  exit()
print("Restoring test set")
pages = restore_evaluation(CATEGORIES)
print("Restoring embeddings")
embeddings, embeddings_size = restore_embeddings(EMBEDDINGS)

graph = tf.Graph()

def matrix_distance(tensor1, tensor2):
  with tf.name_scope("matrix_distance"):
    sub = tf.sub(tensor1, tensor2)
    distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.pow(sub, 2), 1), 1e-10, 1e+37))
    return distance

with graph.as_default():
  # Input

  # Look up embeddings for inputs.
  category_tensor = tf.placeholder(tf.float32, shape=[None, embeddings.shape[1]])
  page_tensor = tf.placeholder(tf.float32, shape=[embeddings.shape[1]])

  category_loss = tf.reduce_mean(matrix_distance(
    category_tensor,
    tf.matmul(tf.ones_like(category_tensor, dtype=tf.float32), tf.diag(page_tensor))
  ))

  # Graph
  loss = 0.0
  counts = 0
  print("Starting calculating pages loss")
  page_proc = 0
  for page in pages.keys():
    if len(pages[page]) == 0:
      continue
    page_proc += 1
    if page_proc % 100000 == 0:
      print("  " + str(page_proc // 100000) + "00k pages parsed")
      print("  Avg loss:", loss/counts)
    category_input = embeddings[np.array(pages[page])]
    page_input = embeddings[page]
    with tf.Session(graph=graph) as session:
      loss += session.run(category_loss, feed_dict={category_tensor:category_input, page_tensor:page_input})
      counts += 1

  print("Average loss: ", loss/counts)