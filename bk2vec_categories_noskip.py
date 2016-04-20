from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import random
import gzip
import csv
import os.path

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

TEXTS = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-text-cleaned.csv.gz'
CATEGORIES = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-page-category-out.csv.gz'
# Borean
#TEXTS = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-text-cleaned.csv.gz'
#CATEGORIES = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-page-category-out.csv.gz'

# Increasing limit for CSV parser
csv.field_size_limit(2147483647)

def wordreader(filename):
  previous = list()
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    try:
      for row in reader:
        if (len(row) is not 2):
          print("Inconsistency in relations file. Previous:", previous, "Current:", row)
          continue
        previous = row
        for word in [row[0]] + row[1].split():
          yield word
    except csv.Error:
      print(u"Dunno why this error happens")


def build_dictionary(reader):
  dictionary = dict()
  reverse_dictionary = dict()
  counts = dict()
  processed = 0
  for word in reader:
    word = str(word)
    processed += 1
    if processed % 100000000 == 0:
      print("  " + str(processed // 100000000) + "00m words parsed (last:", word, ", dic size:", len(dictionary), ")")
    if word in dictionary:
      counts[word] += 1
      continue
    dictionary[word] = len(dictionary)
    reverse_dictionary[dictionary[word]] = word
    counts[word] = 0
  print("Parsing finished")
  return dictionary, reverse_dictionary

# Limiter if there are performance issues
max_pages = 0
max_categories_per_page = 0
test_set = 0.1

def build_pages(filename, dictionary, reverse_dictionary):
  global max_pages, max_categories_per_page, test_set
  pages = dict()
  evaluation = dict()
  maxPagesTitle = "Unknown"
  maxPages = 0
  found = 0
  notfound = 0
  category_found = 0
  category_notfound = 0
  test_set_size = 0
  training_set_size = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    try:
      for row in reader:
        page_title = row[0]
        if page_title not in dictionary:
          notfound += 1
          continue
        found += 1
        if max_pages > 0 and found > max_pages:
          break
        if max_categories_per_page > 0 and len(row) > max_categories_per_page+1:
          row = row[:max_categories_per_page+1]
        if found % 1000000 == 0:
          print("  " + str(found // 1000000) + "m pages parsed")
        page_index = dictionary[page_title]
        if page_index not in pages:
          pages[page_index] = list()
        if page_index not in evaluation:
          evaluation[page_index] = list()
        page_categories = pages[page_index]
        evaluation_current = evaluation[page_index]
        for word in row[1:]:
          if word not in dictionary:
            dictionary[word] = len(dictionary)
            reverse_dictionary[dictionary[word]] = word
            category_notfound += 1
          else:
            category_found += 1
          if test_set > 0 and random.random() <= test_set:
            test_set_size += 1
            evaluation_current.append(dictionary[word])
          else:
            training_set_size += 1
            page_categories.append(dictionary[word])
        if len(page_categories) > maxPages:
          maxPages = len(page_categories)
          maxPagesTitle = page_title
    except csv.Error:
      print(u"Dunno why this error happens")
  print(len(pages), "pages parsed.", "Page with most categories: ", maxPagesTitle, "with", maxPages, "categories")
  print("Training set size:", training_set_size, "Test set size:", test_set_size)
  print("Pages found:", found, "Pages not found:", notfound)
  print("Categories found:", category_found, "Added categories as new tokens:", category_notfound)
  return pages, evaluation


def restore_dictionary(filename):
  dictionary = dict()
  reverse_dictionary = dict()
  processed = 0
  with open(filename + "_dict", 'rb') as csvfile:
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


def store_dictionary(dictionary, filename):
  with open(filename +  "_dict", 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    for value in dictionary.keys():
      writer.writerow([value, dictionary[value]])

def dump_evaluation(evaluation, filename):
  with open(filename + "_test", 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    for value in evaluation.keys():
      writer.writerow([value]+evaluation[value])

if os.path.exists(TEXTS+"_dict"):
  print('Restoring dictionary')
  dictionary, reverse_dictionary = restore_dictionary(TEXTS)
  print('Done')
else:
  dictionary, reverse_dictionary = build_dictionary(wordreader(TEXTS))
  print('Storing dictionary')
  store_dictionary(dictionary, TEXTS)
  print('Done')
vocabulary_size = len(dictionary)
print('Vocabulary size: ', vocabulary_size)
print('Building page -> category dictionary')
pages, evaluation = build_pages(CATEGORIES, dictionary, reverse_dictionary)
vocabulary_size = len(dictionary)
print('Storing test set')
dump_evaluation(evaluation, CATEGORIES)
print('Done')

def word_provider(filename):
  while True:
    for word in wordreader(filename):
      yield word


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
      if buffer[skip_window] not in dictionary:
        print("Word", buffer[skip_window], "is not in dictionary")
        print(buffer)
        print(dictionary[56])
        exit()
      if buffer[target] not in dictionary:
        print("Word", buffer[target], "is not in dictionary")
        print(dictionary[56])
        print(buffer)
        exit()
      batch[i * num_skips + j] = dictionary[buffer[skip_window]]
      labels[i * num_skips + j, 0] = dictionary[buffer[target]]
    buffer.append(reader.next())
  return batch, labels


data_reader = word_provider(TEXTS)
batch, labels = generate_batch(data_reader, dictionary, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])
# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 90  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 8     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample([x for x in range(valid_window)], valid_size))
num_sampled = 64    # Number of negative examples to sample.

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
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name="train_inputs")
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="train_labels")
  train_categories = tf.placeholder(tf.int32, shape=[None], name="train_categories")
  train_category_indexes = tf.placeholder(tf.int32, shape=[None], name="train_category_indexes")

  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="embeddings")
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)), name="NCE_weights")
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="NCE_biases")

  # Recalculating centroids for words in a batch
  with tf.name_scope("category_distances"):
    category_distances = matrix_distance(
      tf.gather(embeddings, train_categories),
      tf.gather(embeddings, train_category_indexes)
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
    category_loss_summary = tf.scalar_summary("category_loss", category_loss)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  with tf.name_scope("skipgram_loss"):
    loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size)
      , name="skipgram_loss"
    )
    skipgram_loss_summary = tf.scalar_summary("skipgram_loss", loss)

  joint_loss = tf.add(loss, category_loss, name="joint_loss")

  # Construct the SGD optimizer using a learning rate of 1.0.
  loss_summary = tf.scalar_summary("joint_loss", joint_loss)
  optimizer = tf.train.GradientDescentOptimizer(1.0, name="joint_objective").minimize(joint_loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 5: Begin training.
num_steps = 2000001

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
  while last_average_loss > 10 and step < num_steps:
    batch_inputs, batch_labels = generate_batch(data_reader, dictionary, batch_size, num_skips, skip_window)
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

    #Calculate category loss once in a while
    #if step % 2000 == 0:
    #  writer.add_summary(session.run(category_loss_summary, feed_dict=feed_dict), step)

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
        average_cat_per_page /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      print("Average categories per batch:", average_cat_per_page)
      last_average_loss = average_loss
      average_loss = 0
      average_cat_per_page = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    #if step % 100000 == 0:
    #  sim = similarity.eval()
    #  for i in xrange(valid_size):
    #    valid_word = reverse_dictionary[valid_examples[i]]
    #    top_k = 8 # number of nearest neighbors
    #    nearest = (-sim[i, :]).argsort()[1:top_k+1]
    #    log_str = "Nearest to %s:" % valid_word
    #    for k in xrange(top_k):
    #      close_word = reverse_dictionary[nearest[k]]
    #      log_str = "%s %s," % (log_str, close_word)
    #    print(log_str)
    step += 1
  print("Retrieving embeddings and normalizing them")
  final_embeddings = normalized_embeddings.eval()
  print("Done")

# Step 6: Dump embeddings to file

with open('embeddings.tsv', 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
  for id, embedding in enumerate(final_embeddings):
    if id not in reverse_dictionary:
      print("Bullshit happened: ", id, embedding)
      continue
    writer.writerow([reverse_dictionary[id]]+[str(value) for value in embedding])

# Step 7: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  filtered = np.nan_to_num(final_embeddings[:plot_only,:])
  low_dim_embs = tsne.fit_transform(filtered)
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn and matplotlib to visualize embeddings.")