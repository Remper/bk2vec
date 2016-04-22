from __future__ import absolute_import
from __future__ import print_function

import collections
import math

from bk2vec.arguments import Arguments
from bk2vec.textreader import *

import numpy as np
import tensorflow as tf


#TEXTS = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-text-cleaned.csv.gz'
#CATEGORIES = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-page-category-out.csv.gz'
# Borean
TEXTS = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-text-cleaned.csv.gz'
CATEGORIES = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-page-category-out.csv.gz'

args = Arguments().show_args().args
text_reader = TextReader(TEXTS)


def dump_evaluation(evaluation, filename):
  with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    for value in evaluation.keys():
      writer.writerow([value]+evaluation[value])

def dump_embeddings(embeddings, reverse_dictionary):
  with open('embeddings-' + str(args.embedding_size) + '-' + get_num_stems_str(args.iterations) + '.tsv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
    for id, embedding in enumerate(embeddings):
      if id not in reverse_dictionary:
        print("Bullshit happened: ", id, embedding)
        continue
      writer.writerow([reverse_dictionary[id]] + [str(value) for value in embedding])

def get_num_stems_str(num_steps):
  letter = ''
  divider = 1
  if num_steps > 1000000:
    letter = 'm'
    divider = 1000000
  elif num_steps > 1000:
    letter = 'k'
    divider = 1000
  if num_steps % divider == 0:
    return str(num_steps // divider)+letter
  else:
    return "{:.1f}".format(float(num_steps) / divider)+letter

dictionary = text_reader.load_dictionary()
pages, evaluation = build_pages(CATEGORIES, dictionary.dict, dictionary.rev_dict)
vocabulary_size = len(dictionary)
print('Vocabulary size: ', vocabulary_size)
print('Storing test and training set')
dump_evaluation(evaluation, "categories_test.tsv")
dump_evaluation(pages, "categories_train.tsv")
print('Done')


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(reader, dictionary, batch_size, num_skips, skip_window):
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(reader.next())
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      input = buffer[skip_window]
      label = buffer[target]
      try:
        input = dictionary.dict[input]
      except:
        input = 0
      try:
        label = dictionary.dict[label]
      except:
        label = 0
      batch[i * num_skips + j] = input
      labels[i * num_skips + j, 0] = label
    buffer.append(reader.next())
  return batch, labels


data_reader = text_reader.endless_words()
batch, labels = generate_batch(data_reader, dictionary, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(dictionary.rev_dict[batch[i]], '->', dictionary.rev_dict[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

def matrix_distance(tensor1, tensor2):
  with tf.name_scope("matrix_distance"):
    sub = tf.sub(tensor1, tensor2)
    distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.pow(sub, 2), 1), 1e-10, 1e+37))
    return distance

def centroid(tensor):
  with tf.name_scope("centroid"):
    return tf.reduce_mean(tensor, 0)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[args.batch_size], name="train_inputs")
  train_labels = tf.placeholder(tf.int32, shape=[args.batch_size, 1], name="train_labels")
  train_categories = tf.placeholder(tf.int32, shape=[None], name="train_categories")
  train_category_indexes = tf.placeholder(tf.int32, shape=[None], name="train_category_indexes")

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, args.embedding_size], -1.0, 1.0), name="embeddings")
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, args.embedding_size],
                            stddev=1.0 / math.sqrt(args.embedding_size)), name="NCE_weights")
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="NCE_biases")

  # Recalculating centroids for words in a batch
  with tf.name_scope("category_distances"):
    category_distances = matrix_distance(
      tf.gather(embeddings, train_categories),
      tf.gather(embeddings, train_category_indexes)
    )
    category_distances = tf.maximum(
      tf.sub(category_distances, tf.constant(1.0)),
      tf.zeros_like(category_distances)
    )


  # Precomputed centroids for each embeddings vector
  # Initialize them with embeddings by default
  # category_centroids = tf.Variable(embeddings.initialized_value())

  # Update centroids
  # tf.scatter_update(category_centroids, train_inputs, recalculated_centroids)

  # Resolving current centroid values
  # current_batch_centroids = tf.nn.embedding_lookup(category_centroids, train_inputs)

  # Categorical knowledge additional term
  with tf.name_scope("category_loss"):
    # Building category objective which is average distance to word centroid
    category_loss = tf.reduce_mean(category_distances)
    category_loss = tf.mul(tf.constant(10.0), category_loss, name="category_contrib_coeff")
    category_loss_summary = tf.scalar_summary("category_loss", category_loss)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  with tf.name_scope("skipgram_loss"):
    loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, args.num_sampled, vocabulary_size)
      , name="skipgram_loss"
    )
    loss = tf.mul(tf.constant(0.1), loss, name="skipgram_contrib_coeff")
    skipgram_loss_summary = tf.scalar_summary("skipgram_loss", loss)

  joint_loss = tf.add(loss, category_loss, name="joint_loss")

  # Construct the SGD optimizer using a learning rate of 1.0.
  loss_summary = tf.scalar_summary("joint_loss", joint_loss)
  optimizer = tf.train.GradientDescentOptimizer(1.0, name="joint_objective").minimize(joint_loss)

  # Normalize final embeddings
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

# Step 5: Begin training.
with tf.Session(graph=graph) as session:
  # merged = tf.merge_all_summaries()
  category_merged = tf.merge_summary([skipgram_loss_summary, loss_summary, category_loss_summary])
  writer = tf.train.SummaryWriter("logs", graph)
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  average_cat_per_page = 0
  last_average_loss = 241
  step = 0
  while step < args.iterations:
    batch_inputs, batch_labels = generate_batch(data_reader, dictionary, args.batch_size, args.num_skips, args.window_size)
    categories = list()
    category_indexes = list()
    for id, input in enumerate(batch_inputs):
      if input in pages:
        for i in pages[input]:
          category_indexes.append(id)
          categories.append(i)
    if len(categories) is 0:
      categories.append(0)
      category_indexes.append(1)
    average_cat_per_page += len(categories)
    feed_dict = {
      train_inputs: batch_inputs, train_labels: batch_labels,
      train_categories: categories, train_category_indexes: category_indexes
    }

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()

    summary, _, loss_val = session.run([category_merged, optimizer, joint_loss], feed_dict=feed_dict)
    writer.add_summary(summary, step)

    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
        average_cat_per_page /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", get_num_stems_str(step), ": ", average_loss)
      print("Average categories per batch:", average_cat_per_page)
      last_average_loss = average_loss
      average_loss = 0
      average_cat_per_page = 0

    step += 1
  print("Retrieving embeddings and normalizing them")
  final_embeddings = normalized_embeddings.eval()
  print("Done")

# Step 6: Dump embeddings to file
dump_embeddings(final_embeddings, dictionary.rev_dict)