# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import tensorflow as tf
import numpy as np

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
      index = int(row[0])
      pages[index] = np.array(map(int, row[1:]))
      categories += len(pages[index])
  print("Loaded", len(pages.values()), "pages with", categories, "categories")
  return pages


def restore_embeddings(filename):
  embeddings = list()
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
        if embeddings_count % 1000000 == 0:
          print("  "+str(embeddings_count // 1000000)+"m words parsed")
    except csv.Error:
      print(u"Dunno why this error happens")
  return embeddings, embeddings_size

print("Restoring embeddings")
embeddings, embeddings_size = restore_embeddings(EMBEDDINGS)
pages = restore_evaluation(CATEGORIES)
print(len(embeddings), "loaded of", embeddings_size, "dimensions")

graph = tf.Graph()

def matrix_distance(tensor1, tensor2):
  with tf.name_scope("matrix_distance"):
    sub = tf.sub(tensor1, tensor2)
    distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.pow(sub, 2), 1), 1e-10, 1e+37))
    return distance

with graph.as_default():
  # Input

  # Look up embeddings for inputs.
  embeddings_input = tf.constant(embeddings)

  # Map relations
  categories_input = list()
  page_input = list()
  for page in pages:
    category_tensor = tf.constant(pages[page], name="page_input")
    categories_input.append(category_tensor)
    page_input.append(page)

  # Graph
  categories_distance = list()
  for id, categories in enumerate(categories_input):
    resolved_categories = tf.gather(embeddings, categories)
    categories_distance.append(tf.reduce_mean(matrix_distance(
      resolved_categories,
      tf.matmul(tf.ones_like(resolved_categories), tf.diag(tf.gather(embeddings, pages[id])))
    )))
  category_loss = tf.reduce_mean(tf.pack(categories_distance))

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  loss = session.run(category_loss, feed_dict=dict())
  print("Average loss: ", loss)