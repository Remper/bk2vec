# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import tensorflow as tf

#DICTIONARY = '/Users/remper/Projects/bk2vec/borean/embeddings_80_500k_traintest_dict.gz'
#CATEGORIES = '/Users/remper/Projects/bk2vec/borean/embeddings_80_500k_traintest_test.gz'
#EMBEDDINGS = '/Users/remper/Projects/bk2vec/borean/embeddings_80_500k_traintest.tsv.gz'
# Borean
#DICTIONARY = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-text-cleaned.csv.gz_dict'
#CATEGORIES = '/borean1/data/pokedem-experiments/bk2vec/embeddings_80_500k_traintest_test.gz'
#EMBEDDINGS = '/borean1/data/pokedem-experiments/bk2vec/embeddings_80_500k_traintest.tsv.gz'
# Auster
DICTIONARY = 'embeddings_80_500k_traintest_dict.gz'
CATEGORIES = 'embeddings_80_500k_traintest_test.gz'
EMBEDDINGS = 'embeddings_80_500k_traintest.tsv.gz'

def restore_dictionary(filename):
  dictionary = dict()
  reverse_dictionary = dict()
  processed = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    for row in reader:
      row[0] = str(row[0])
      row[1] = int(row[1])
      processed += 1
      if processed % 3000000 == 0:
        print("  " + str(processed // 1000000) + "m words parsed")
      dictionary[row[0]] = row[1]
      reverse_dictionary[row[1]] = row[0]
  return dictionary, reverse_dictionary


def restore_evaluation(filename):
  pages = dict()
  categories = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    for row in reader:
      if row[0] not in pages:
        pages[row[0]] = list()
      categories = pages[row[0]]
      categories.extend(row[1:])
      categories += len(row[1:])
  print("Loaded", len(pages.values()), "pages with", categories, "categories")
  return pages


def restore_embeddings(dictionary, filename):
  embeddings = list()
  embeddings_size = 0
  embeddings_count = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    try:
      for row in reader:
        if embeddings_size == 0:
          embeddings_size = len(row) - 1
        if row[0] not in dictionary:
          print("Word", row[0], "haven't been found in a dictionary")
          continue
        dict_index = dictionary[row[0]]
        if dict_index != len(embeddings):
          print("Inconsistency dictionary -> embeddings, pages might be off", dict_index, len(embeddings))
          exit()
        embeddings_count += 1
        embedding = map(float, row[1:])
        embeddings.append(embedding)
        if embeddings_count % 1000000 == 0:
          print("  "+str(embeddings_count // 1000000)+"m words parsed")
    except csv.Error:
      print(u"Dunno why this error happens")
  return embeddings, embeddings_size


print("Restoring dictionary")
dictionary, reverse_dictionary = restore_dictionary(DICTIONARY)
print("Dictionary restored")
print("Restoring embeddings")
embeddings, embeddings_size = restore_embeddings(dictionary, EMBEDDINGS)
pages = restore_evaluation(CATEGORIES)
print(len(embeddings), "loaded of", embeddings_size, "dimensions")
print("Sample embedding: yaroslav", embeddings[dictionary['yaroslav']])

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