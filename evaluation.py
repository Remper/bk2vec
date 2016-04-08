# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import zipfile
import csv
import tensorflow as tf

def read_relations(filename):
  f = zipfile.ZipFile(filename)
  reverse_dict = {}
  dict = {}
  previous = []
  for name in f.namelist():
    for line in f.open(name, "r").readlines():
      relation = line.strip().split("\t")
      if len(relation) is not 2:
        print("Inconsistency in relations file. Previous:", previous, "Current:", relation)
        continue
      previous = relation
      relation[1] = relation[1].split("-")[0].lower()
      if relation[1] is 'us':
        relation[1] = 'united_states'
      if relation[1] not in reverse_dict:
        reverse_dict[relation[1]] = []
      reverse_dict[relation[1]].append(relation[0])
      dict[relation[0]] = relation[1]
  return reverse_dict, dict

def read_embeddings(filename):
  f = zipfile.ZipFile(filename)
  dictionary = dict()
  embeddings = list()
  embeddings_size = 0
  for name in f.namelist():
    with f.open(name, 'rU') as csvfile:
      reader = csv.reader(csvfile, delimiter='\t')
      try:
        for row in reader:
          if embeddings_size == 0:
            embeddings_size = len(row) - 1
          if row[0] in dictionary:
            continue
          dictionary[row[0]] = len(embeddings)
          embedding = map(float, row[1:])
          embeddings.append(embedding)
          if len(embeddings) % 20000 == 0:
            print("  " + str(len(embeddings) / 1000) + "k words parsed")
      except csv.Error:
        print(u"Dunno why this error happens")
  return embeddings, dictionary, embeddings_size

embeddings, dictionary, embeddings_size = read_embeddings("/Users/remper/Google Drive/bk2vec/experiments/simple_relations/clean_embeddings.zip")
print(len(embeddings), "loaded of", embeddings_size, "dimensions")
print("Sample embedding: yaroslav", embeddings[dictionary['yaroslav']])
relations, relations_dict = read_relations("/Users/remper/Google Drive/bk2vec/experiments/simple_relations/relations.zip")
print(len(relations_dict),"loaded of", len(relations), "categories")

graph = tf.Graph()

with graph.as_default():
  # Input

  # Look up embeddings for inputs.
  embeddings_input = tf.constant(embeddings)

  # Map relations
  categories_input = list()
  missing_words = list()
  for category in relations.keys():
    category_list = list()
    for word in relations[category]:
      if word not in dictionary:
        missing_words.append(word)
        word = 'UNK'
      category_list.append(dictionary[word])
    category_tensor = tf.constant(category_list, name="category_input")
    categories_input.append(category_tensor)
  print("Words missing in embeddings ("+str(len(missing_words))+"):",missing_words)

  # Graph

  # Look up embeddings for categories
  categories = list()
  for category in categories_input:
    categories.append(tf.nn.embedding_lookup(embeddings_input, category, name="category_embedding"))

  # Categorical knowledge additional term
  with tf.name_scope("category_loss") as scope:
    losses = list()
    for category in categories:
      centroid = tf.matmul(tf.ones_like(category), tf.diag(tf.reduce_mean(category, 0)), name="centroid")
      losses.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
        category, centroid), 2), 1), name="category_sqrt"), name="mean_loss_in_category"))

    # Building join objective, normalizing second objective by the amount of categories
    category_loss = tf.add_n(losses)

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  loss = session.run(category_loss, feed_dict=dict())
  print("Average loss: ", loss)