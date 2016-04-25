from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import time
from threading import Thread

from bk2vec.arguments import Arguments
from bk2vec.embeddings import Embeddings
from bk2vec.textreader import *

import numpy as np
import tensorflow as tf

# TEXTS = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-text-cleaned.csv.gz'
# CATEGORIES = '/Users/remper/Downloads/bk2vec_input/enwiki-20140203-page-category-out.csv.gz'
# Borean
TEXTS = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-text-cleaned.csv.gz'
CATEGORIES = '/borean1/data/pokedem-experiments/bk2vec/alessio/enwiki-20140203-page-category-out.csv.gz'

args = Arguments().show_args().args
text_reader = TextReader(TEXTS)


class EvaluationDumper(Thread):
    def __init__(self, evaluation, postfix):
        Thread.__init__(self)
        self._evaluation = evaluation
        self.postfix = postfix

    def run(self):
        count = 0
        with gzip.open('categories-' + self.postfix + '.tsv.gz', 'wb') as writer:
            for value in self._evaluation.keys():
                count += 1
                row = self._evaluation[value]
                if len(row) == 0:
                    continue
                writer.write(str(value))
                writer.write('\t')
                writer.write('\t'.join(map(str, row)))
                writer.write('\n')
                if count % 100000 == 0:
                    print("  ", str(count // 1000) + "k words parsed (" + self.postfix + ")")
        print("Finished dumping ", self.postfix)
        del self._evaluation


def dump_evaluation(evaluation, postfix):
    dumper = EvaluationDumper(evaluation, postfix)
    dumper.start()
    return dumper


def control_evaluation(threads):
    for dumper in threads:
        dumper.join()


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
        return str(num_steps // divider) + letter
    else:
        return "{:.1f}".format(float(num_steps) / divider) + letter


dictionary = text_reader.load_dictionary()
tasks = list()
pages = dict()
if not args.clean:
    pages, evaluation = build_pages(CATEGORIES, dictionary.dict, dictionary.rev_dict)
    print('Storing test and training set')
    tasks.append(dump_evaluation(evaluation, "test"))
    tasks.append(dump_evaluation(pages, "train"))
    del evaluation
else:
    print('Ignoring categories file: clean embeddings requested')
vocabulary_size = len(dictionary)
print('Vocabulary size: ', vocabulary_size)


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(reader, dictionary, batch_size, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(reader.next())
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
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


class Pages:
    def __init__(self, pages):
        self._pages = pages

    def _get_next_page(self):
        while True:
            for page in self._pages.keys():
                yield page

    def generate_detached_batch(self, batch_size):
        batch_indexes = list()
        batch = list()
        while len(batch_indexes) < batch_size:
            page = self._get_next_page()
            for ele in self._pages[page]:
                batch_indexes.append(page)
                batch.append(ele)
        return batch_indexes, batch

    def generate_batch(self, input_batch):
        batch_indexes = list()
        batch = list()
        for page in input_batch:
            if page in self._pages:
                for ele in self._pages[input]:
                    batch_indexes.append(page)
                    batch.append(ele)
        return batch_indexes, batch

data_reader = text_reader.endless_words()
batch, labels = generate_batch(data_reader, dictionary, batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], '->', labels[i, 0])
    print(dictionary.rev_dict[batch[i]], '->', dictionary.rev_dict[labels[i, 0]])


# Step 4: Build and train a skip-gram model.

def matrix_distance(tensor1, tensor2):
    with tf.name_scope("matrix_distance"):
        sub = tf.sub(tensor1, tensor2)
        distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.pow(sub, 2), 1), 1e-10, 1.0))
        return distance


def centroid(tensor):
    with tf.name_scope("centroid"):
        return tf.reduce_mean(tensor, 0)


graph = tf.Graph()

with graph.as_default():
    embeddings = Embeddings(vocabulary_size, args.embeddings_size)

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[args.batch_size], name="train_inputs")
    train_labels = tf.placeholder(tf.int32, shape=[args.batch_size, 1], name="train_labels")
    train_categories = tf.placeholder(tf.int32, shape=[None], name="train_categories")
    train_category_indexes = tf.placeholder(tf.int32, shape=[None], name="train_category_indexes")

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings.tensor(), train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([embeddings.vocabulary_size, embeddings.size],
                                stddev=1.0 / math.sqrt(embeddings.size)), name="NCE_weights")
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="NCE_biases")

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    with tf.name_scope("skipgram_loss"):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, args.num_sampled, embeddings.vocabulary_size)
            , name="skipgram_loss"
        )
        loss = tf.mul(tf.constant(0.1), loss, name="skipgram_contrib_coeff")
        skipgram_loss_summary = tf.scalar_summary("skipgram_loss", loss)

    joint_loss = loss

    if not args.clean:
        # Recalculating centroids for words in a batch
        with tf.name_scope("category_distances"):
            # Calculating distances towards category tokens
            category_distances = matrix_distance(
                tf.gather(embeddings.tensor(), train_categories),
                tf.gather(embeddings.tensor(), train_category_indexes)
            )
            # Margin (we don't want categories to be squashed into the single dot)
            category_distances = tf.maximum(
                tf.sub(category_distances, tf.constant(0.5)),
                tf.zeros_like(category_distances)
            )

        # Categorical knowledge additional term
        with tf.name_scope("category_loss"):
            # Building category objective which is average distance to word centroid
            category_loss = tf.reduce_mean(category_distances)
            category_loss = tf.mul(tf.constant(10.0), category_loss, name="category_contrib_coeff")
            category_loss_summary = tf.scalar_summary("category_loss", category_loss)

        joint_loss = tf.add(loss, category_loss, name="joint_loss")

    # Construct the SGD optimizer using a learning rate of 1.0.
    loss_summary = tf.scalar_summary("joint_loss", joint_loss)
    optimizer = tf.train.GradientDescentOptimizer(1.0, name="joint_objective").minimize(joint_loss)

# Step 5: Begin training.
with tf.Session(graph=graph) as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs", graph)
    tf.initialize_all_variables().run()
    print("Initialized")

    average_loss = 0
    average_cat_per_page = 0
    last_average_loss = 241
    step = 0
    timestamp = time.time()
    pages = Pages(pages)
    while step < args.iterations:
        batch_inputs, batch_labels = generate_batch(data_reader, dictionary, args.batch_size, args.num_skips,
                                                    args.window_size)
        categories = list()
        category_indexes = list()
        if not args.clean:
            if args.detached:
                category_indexes, categories = pages.generate_detached_batch(args.batch_size)
            else:
                category_indexes, categories = pages.generate_batch(batch_inputs)
        if len(categories) is 0:
            categories.append(0)
            category_indexes.append(0)
        average_cat_per_page += len(categories)
        feed_dict = {
            train_inputs: batch_inputs, train_labels: batch_labels,
            train_categories: categories, train_category_indexes: category_indexes
        }

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()

        summary, _, loss_val = session.run([merged, optimizer, joint_loss], feed_dict=feed_dict)
        writer.add_summary(summary, step)

        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                average_cat_per_page /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", get_num_stems_str(step), ": ", average_loss, "(", time.time() - timestamp,
                  "s)")
            timestamp = time.time()
            print("Average categories per batch:", average_cat_per_page)
            last_average_loss = average_loss
            average_loss = 0
            average_cat_per_page = 0

        step += 1

    del pages
    timestamp = time.time()
    norm_tensor = embeddings.normalized_tensor().eval()
    print("Tensor normalized in", time.time() - timestamp, "s")

# Step 6: Dump embeddings to file
print("Dumping embeddings on disk")
embeddings.dump(
    'embeddings-' + str(embeddings.size) + '-' + get_num_stems_str(args.iterations) + '.tsv',
    norm_tensor,
    dictionary
)

print("Waiting for evaluation dumping to finish...")
control_evaluation(tasks)
print("Done")
