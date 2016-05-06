from __future__ import absolute_import
from __future__ import print_function

import math
import os

from params import TEXTS
from params import CATEGORIES_NOTOKEN
from params import CATEGORIES_TOKEN

from bk2vec.arguments import Arguments
from bk2vec.embeddings import Embeddings
from bk2vec.tftextreader import *
from bk2vec.textreader import build_pages
from bk2vec.evaluation import *
from bk2vec.utils import Log

import tensorflow as tf

args = Arguments().show_args().args
if args.threads < 2:
    print("Can't start with a single thread")
pre_thread_count = int(args.threads * 0.25)
if pre_thread_count == 0:
    pre_thread_count = 1
proc_threads = args.threads - pre_thread_count
text_reader = TextReader(args.output+'text.bin', args.window_size, pre_thread_count, args.batch_size)

log = Log(args)

if len(args.output) > 0:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        log.print("Created output directory")

if args.notoken:
    CATEGORIES = CATEGORIES_NOTOKEN
else:
    CATEGORIES = CATEGORIES_TOKEN

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

if not os.path.exists(text_reader.filename):
    dictionary = text_reader.text2bin(TEXTS)
else:
    dictionary = text_reader.restore_dictionary()

tasks = list()
pages, evaluation = build_pages(CATEGORIES, dictionary.dict, dictionary.rev_dict)
log.print('Started storing test and training set')
tasks.append(dump_evaluation(evaluation, "test", folder=args.output))
tasks.append(dump_evaluation(pages, "train", folder=args.output))
del evaluation
vocabulary_size = len(dictionary)
log.print('Vocabulary size: ', vocabulary_size)


class Pages:
    def __init__(self, pages):
        self._pages = pages
        self._generator = self._get_next_page()

    def _get_next_page(self):
        while True:
            for page in self._pages.keys():
                yield page

    def generate_detached_batch(self, batch_size):
        batch_indexes = list()
        batch = list()
        while len(batch_indexes) < batch_size:
            page = self._generator.next()
            for ele in self._pages[page]:
                batch_indexes.append(page)
                batch.append(ele)
        return batch_indexes, batch

    def generate_batch(self, input_batch, target_batch):
        batch_indexes = list()
        batch = list()
        for page in input_batch+target_batch.flatten():
            if page in self._pages:
                for ele in self._pages[page]:
                    batch_indexes.append(page)
                    batch.append(ele)
        return batch_indexes, batch


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
    embeddings = Embeddings(vocabulary_size, args.embeddings_size)

    # Input data.
    train_inputs, train_labels = text_reader.get_reading_ops()
    train_inputs = tf.squeeze(train_inputs)
    train_categories = tf.placeholder(tf.int32, shape=[None], name="train_categories")
    train_category_indexes = tf.placeholder(tf.int32, shape=[None], name="train_category_indexes")
    global_step = tf.Variable(0, name='global_step', trainable=False)

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
        #loss = tf.mul(tf.constant(0.1), loss, name="skipgram_contrib_coeff")
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
            if args.margin > 0:
                with tf.name_scope("margin_cutoff"):
                    category_distances = tf.maximum(
                        tf.sub(category_distances, tf.constant(args.margin, dtype=tf.float32)),
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
    optimizer = tf.train.GradientDescentOptimizer(1.0, name="joint_objective").minimize(joint_loss, global_step=global_step)
    merged = tf.merge_all_summaries()


class TrainingWorker(Thread):
    def __init__(self, iterations, session, writer):
        Thread.__init__(self)
        self._iterations = iterations
        self._session = session
        self._writer = writer

    def run(self):
        step = global_step.eval(self._session)
        average_loss = 0
        count = 0
        while step < self._iterations:
            categories = [0]
            category_indexes = [0]
            feed_dict = {train_categories: categories, train_category_indexes: category_indexes}
            step, summary, _, loss_val = self._session.run(
                [global_step, merged, optimizer, joint_loss],
                feed_dict=feed_dict
            )
            average_loss += loss_val
            count += 1
            writer.add_summary(summary, step)

            if count % 2000 == 0:
                average_loss /= 2000
                log.print("Average loss at step "+get_num_stems_str(step)+":", average_loss)
                average_loss = 0

# Step 5: Begin training.
with tf.Session(graph=graph) as session:
    timestamp = time.time()
    writer = tf.train.SummaryWriter("logs", graph)
    tf.initialize_all_variables().run()
    log.print("Initialized", ("%.5f" % (time.time() - timestamp)), "s")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    step = 0
    pages = Pages(pages)
    workers = list()
    for _ in range(proc_threads):
        worker = TrainingWorker(args.iterations, session, writer)
        worker.start()
        workers.append(worker)
    log.print("Started", proc_threads, "worker threads")

    while len(workers) > 0:
        filtered = list()
        for worker in workers:
            if worker.is_alive():
                filtered.append(worker)
        workers = filtered
        time.sleep(20)
        newstep = global_step.eval(session)
        log.print(newstep, "steps processed ("+str((newstep-step)/20)+" steps/s)")
        step = newstep

    coord.request_stop()
    coord.join(threads)

    del pages
    timestamp = time.time()
    norm_tensor = embeddings.normalized_tensor().eval()
    log.print("Tensor normalized in", ("%.5f" % (time.time() - timestamp)), "s")

# Step 6: Dump embeddings to file
log.print("Dumping embeddings on disk")
embeddings.dump(
    args.output+'embeddings-' + str(embeddings.size) + '-' + get_num_stems_str(args.iterations) + '.tsv',
    norm_tensor,
    dictionary
)

log.print("Waiting for evaluation dumping to finish...")
control_evaluation(tasks)
log.print("Done")
log.close()