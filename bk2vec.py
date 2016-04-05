from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.
def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name).decode("utf8").split()
  f.close()

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

words = read_data('enwiki-20160305-text.csv.zip')+read_data(filename)
relations_dict, relations = read_relations('relations.zip')
print('Data size', len(words))
print('Relations size', len(relations), len(relations_dict.keys()))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 200000

def build_dataset(words, relations):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  data = list()
  unk_count = 0
  categories = list()
  matches = 0

  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      if word in relations:
        dictionary[word] = len(dictionary)
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
    data.append(index)
    if word in relations:
      matches += 1
      categories.append(relations[word])
    else:
      categories.append(index)
    if len(data) % 10000000 == 0:
      print("  "+str(len(data)/1000000)+"m words parsed")
  print('Relation matches: '+str(matches))
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary, categories

data, count, dictionary, reverse_dictionary, categories = build_dataset(words, relations)
vocabulary_size = len(dictionary)
del words  # Hint to reduce memory.
# Building input list of relations ready to lookup at training time
relations_dict_idx = dict()
for words in relations_dict.keys():
  sublist = list()
  for word in relations_dict[words]:
    if word in dictionary:
      sublist.append(dictionary[word])
  if (len(sublist) > 0):
    print("Category \""+words+"\" found "+str(len(sublist))+"/"+str(len(relations_dict[words]))+" words")
    relations_dict_idx[words] = sublist
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample([x for x in range(valid_window)], valid_size))
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  categories_input = list()

  for category in relations_dict_idx.keys():
    category_tensor = tf.constant(relations_dict_idx[category], name="category_input")
    categories_input.append(category_tensor)
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Look up embeddings for categories
    categories = list()
    for category in categories_input:
      categories.append(tf.nn.embedding_lookup(embeddings, category, name="category_embedding"))

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Categorical knowledge additional term
  losses = list()
  for category in categories:
    centroid = tf.matmul(tf.ones_like(category), tf.diag(tf.reduce_mean(category, 0)), name="centroid")
    losses.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
      category, centroid), 2), 1), name="category_sqrt"), name="mean_loss_in_category"))


  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                   num_sampled, vocabulary_size), name="skipgram_loss")

  # Building join objective, normalizing second objective by the amount of categories
  category_loss = tf.mul(tf.constant(0.1),tf.add_n(losses), name="category_loss")
  loss = tf.add(loss, category_loss, name="joint_loss")

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0, name="joint_objective").minimize(loss)
  #optimizer_category = tf.train.GradientDescentOptimizer(1.0, name="Category_loss").minimize(category_loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.train.SummaryWriter("logs", session.graph_def)
  # We must initialize all variables before we use them.
  tf.initialize_all_variables().run()
  print("Initialized")

  average_loss = 0
  average_loss2 = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels   = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    # Second objective is being minimized only once per 20 batches
    #if step % 20 == 0:
      #__, loss_val2 = session.run([optimizer_category, category_loss], feed_dict=feed_dict)
      #average_loss2 += loss_val2
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      #print("Average category loss at step ", step, ": ", average_loss2)
      average_loss = 0
      average_loss2 = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.

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