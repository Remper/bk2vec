from __future__ import absolute_import
from __future__ import print_function

from params import TEXTS
from params import CATEGORIES_NOTOKEN
from params import CATEGORIES_TOKEN

from bk2vec.evaluation import *
from bk2vec.arguments import Arguments
from bk2vec.embeddings import Embeddings
from bk2vec.relations import Relations
from bk2vec.tftextreader import *
from bk2vec.utils import *
from bk2vec.nce_loss import nce_loss

import math
import os
import tensorflow as tf

args = Arguments().show_args().args
if args.threads < 2:
    print("Can't start with a single thread")
pre_thread_count = int(args.threads * 0.25)
if pre_thread_count == 0:
    pre_thread_count = 1
proc_threads = args.threads - pre_thread_count
text_reader = TextReader(args.output+'text.bin', args.window_size, pre_thread_count, args.batch_size)
tf.set_random_seed(args.seed)

log = Log(args)

if len(args.output) > 0:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        log.print("Created output directory")

if args.notoken:
    CATEGORIES = CATEGORIES_NOTOKEN
else:
    CATEGORIES = CATEGORIES_TOKEN

if not os.path.exists(text_reader.filename):
    dictionary = text_reader.text2bin(TEXTS, blacklist=args.blacklist)
else:
    dictionary = text_reader.restore_dictionary()


relations = Relations(dictionary)
if not args.clean:
    text_reader.set_relations(relations)
    text_reader.enable_categories()
    timestamp = time.time()
    old_vocabulary_size = len(dictionary)
    relations.load_categories(CATEGORIES)
    if not args.norelations:
        text_reader.enable_relations()
        relations.load_relations()
    vocabulary_size = len(dictionary)
    log.print('Knowledge loaded ('+('%.2f' % (time.time() - timestamp))+'s)')
    log.print('  Unique relations loaded: ', relations.get_num_relations())
    log.print('  Words with categories: ', relations.get_num_words_with_categories())

    difference = vocabulary_size - old_vocabulary_size
    if difference > 0:
        log.print('Added', difference, 'words. Updating dictionary:')
        text_reader.store_dictionary(dictionary)
vocabulary_size = len(dictionary)
log.print('Vocabulary size: ', vocabulary_size)

wordsim353 = WordSimilarity.wordsim353("datasets/combined.csv", dictionary.dict)
simlex999 = WordSimilarity.simlex999("datasets/SimLex-999.txt", dictionary.dict)
log.print("Similarity pairs loaded")
analogy = Analogy.from_file("datasets/questions-words.txt", dictionary.dict)
log.print("Analogy entries loaded")

# Step 4: Build and train a skip-gram model.


def matrix_distance(tensor1, tensor2):
    with tf.name_scope("matrix_distance"):
        diff = tf.squared_difference(tensor1, tensor2)
        distance = tf.sqrt(tf.clip_by_value(tf.reduce_sum(diff, 1), 1e-10, 1e+37))
        return distance


def cosine_matrix_distance(tensor1, tensor2):
    with tf.name_scope("cosine_matrix_distance"):
        tensor1 = tf.nn.l2_normalize(tensor1, dim=1)
        tensor2 = tf.nn.l2_normalize(tensor2, dim=1)
        result = tf.abs(tf.matmul(tensor1, tensor2, transpose_b=True))
        return tf.sub(tf.ones_like(result), result)


def centroid(tensor):
    with tf.name_scope("centroid"):
        return tf.reduce_mean(tensor, 0)


graph = tf.Graph()

with graph.as_default():
    embeddings = Embeddings(vocabulary_size, args.embeddings_size)
    wordsim353.register_op(embeddings.tensor())
    simlex999.register_op(embeddings.tensor())
    analogy.register_op(embeddings.tensor())

    # Input data.
    train_inputs, train_labels = text_reader.get_reading_ops()
    train_inputs = tf.squeeze(train_inputs)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Look up embeddings for inputs.
    embed = tf.gather(embeddings.tensor(), train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal(
            [embeddings.vocabulary_size, embeddings.size],
            stddev=1.0 / math.sqrt(embeddings.size)
        ), name="NCE_weights")
    nce_biases = tf.Variable(
        tf.truncated_normal(
            [embeddings.vocabulary_size],
            stddev=1.0 / math.sqrt(embeddings.vocabulary_size)
        ), name="NCE_biases")


    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    with tf.name_scope("skipgram_loss"):
        loss = tf.reduce_mean(
            nce_loss(nce_weights, nce_biases, embed, train_labels, args.num_sampled, embeddings.vocabulary_size)
            , name="skipgram_loss"
        )
        #loss = tf.mul(tf.constant(0.1), loss, name="skipgram_contrib_coeff")
        skipgram_loss_summary = tf.scalar_summary("skipgram_loss", loss)

    joint_loss = loss

    category_size = tf.constant(0)
    relation_size = tf.constant(0)
    if not args.clean:
        losses = [loss]

        with tf.name_scope("category_distances"):
            # Calculating distances towards category tokens
            categorical_batch = text_reader.get_category_ops()
            category_distances = matrix_distance(
                tf.gather(embeddings.tensor(), categorical_batch[:, 0]),
                tf.gather(embeddings.tensor(), categorical_batch[:, 1])
            )
            # Margin (we don't want categories to be squashed into the single dot)
            if args.margin > 0:
                with tf.name_scope("margin_cutoff"):
                    category_distances = tf.abs(
                        tf.sub(category_distances, tf.constant(args.margin, dtype=tf.float32))
                    )
            category_size = tf.size(category_distances)

        # Categorical knowledge additional term
        with tf.name_scope("categorical_loss"):
            # Building category objective which is average distance to word centroid
            category_loss = tf.reduce_mean(category_distances)
            # category_loss = tf.mul(tf.constant(10.0), category_loss, name="category_contrib_coeff")
            category_loss_summary = tf.scalar_summary("category_loss", category_loss)
        losses.append(category_loss)

        if not args.norelations:

            with tf.name_scope("relation_distances"):
                relational_batch = text_reader.get_relations_ops()
                t_rel_word1 = tf.gather(embeddings.tensor(), relational_batch[:, 0])
                t_relations = tf.gather(embeddings.tensor(), relational_batch[:, 1])
                t_rel_word2 = tf.gather(embeddings.tensor(), relational_batch[:, 2])
                t_rel_cor_word1 = tf.gather(embeddings.tensor(), relational_batch[:, 3])
                t_rel_cor_word2 = tf.gather(embeddings.tensor(), relational_batch[:, 4])

                true_relation_distances = matrix_distance(
                    tf.add(t_rel_word1, t_relations),
                    t_rel_word2
                )
                false_relation_distances = matrix_distance(
                    tf.add(t_rel_cor_word1, t_relations),
                    t_rel_cor_word2
                )
                relation_distances = tf.abs(tf.add(
                    tf.sub(true_relation_distances, false_relation_distances),
                    tf.constant(1.0)
                ))
                relation_size = tf.size(relation_distances)

            # Relational knowledge additional term
            with tf.name_scope("relational_loss"):
                # Building category objective which is average distance to word centroid
                relational_loss = tf.reduce_mean(relation_distances)
                relational_loss += tf.contrib.layers.l2_regularizer(0.01)(embed)
                # relational_loss = tf.mul(tf.constant(10.0), relational_loss, name="category_contrib_coeff")
                relational_loss_summary = tf.scalar_summary("relational_loss", relational_loss)
            losses.append(relational_loss)

        joint_loss = tf.add_n(losses, name="joint_loss")
    joint_loss = tf.clip_by_value(joint_loss, 1e-10, args.num_sampled*10)

    # Construct the SGD optimizer using a learning rate of 1.0.
    loss_summary = tf.scalar_summary("joint_loss", joint_loss)
    learning_rate = 1.0
    #learning_rate = tf.train.exponential_decay(1.0, global_step, 150000, 0.9, staircase=True)
    optimizer = tf.train.AdagradOptimizer(learning_rate, use_locking=True, name="joint_objective").minimize(joint_loss, global_step=global_step)
    merged = tf.merge_all_summaries()
    embeddings.normalized_tensor()


class TrainingWorker(Thread):
    def __init__(self, idx, iterations, session, writer):
        Thread.__init__(self)
        self._idx = idx
        self._iterations = iterations
        self._session = session
        self._writer = writer

    def run(self):
        step = global_step.eval(self._session)
        average_loss = 0
        average_cats = 0
        average_rels = 0
        count = 0
        last_count = 0
        while step < self._iterations:
            step, summary, _, loss_val, cur_cat_size, cur_rel_size = self._session.run(
                [global_step, merged, optimizer, joint_loss, category_size, relation_size]
            )
            average_loss += loss_val
            average_cats += cur_cat_size
            average_rels += cur_rel_size
            count += 1
            if self._iterations > 5000000:
                if count % int(math.log(self._iterations)) == 0:
                    writer.add_summary(summary, step)
            else:
                writer.add_summary(summary, step)

            if count % 2000 == 0 and (count // 2000) % proc_threads == self._idx:
                average_loss /= count - last_count
                average_cats /= count - last_count
                average_rels /= count - last_count
                log.print("[Worker "+str(self._idx)+"] Average loss at step "+get_num_stems_str(step)+":",
                          "%.5f" % average_loss,
                          "(avg cats:", ("%.2f" % average_cats)+", avg rels:", ("%.2f" % average_rels)+")")
                last_count = count
                average_loss = 0
                average_rels = 0
                average_cats = 0

# Step 5: Begin training.
tasks = []
with tf.Session(graph=graph) as session:
    timestamp = time.time()
    writer = tf.train.SummaryWriter("logs", graph)
    tf.initialize_all_variables().run()
    log.print("Initialized", ("%.5f" % (time.time() - timestamp)), "s")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    step = 0
    workers = list()
    for idx in range(proc_threads):
        worker = TrainingWorker(idx, args.iterations, session, writer)
        worker.start()
        workers.append(worker)
    log.print("Started", proc_threads, "worker threads")

    show = 0
    timeadj = 0
    while len(workers) > 0:
        filtered = list()
        for worker in workers:
            if worker.is_alive():
                filtered.append(worker)
        workers = filtered
        time.sleep(20)
        newstep = global_step.eval(session)
        log.print(newstep,
                  "steps processed ("+str(int(float(newstep-step)/(20+timeadj)))+" steps/s,",
                  get_num_stems_str(text_reader.get_reader_stats(session)), "words)")
        step = newstep

        show += 1
        if show % 10 == 0:
            debug_results = session.run(text_reader.get_debug_op())
            batch, examples, filtered_examples = debug_results[:3]
            if len(debug_results) > 3:
                relations_debug = debug_results[3]
                total = relations_debug.shape[0]
                num = 5
                if total < num:
                    num = total
                print("Relations (showing %d out of %d):" % (num, total))
                for idx in range(num):
                    debug_example = [dictionary.rev_dict[ele] for ele in relations_debug[idx]]
                    print(debug_example[0], '-(' + debug_example[1] + ')>', debug_example[2], "(Corrupted:",
                          debug_example[3], '-(' + debug_example[1] + ')>', debug_example[4] + ')')
            print("Batch resolved: ", " ".join([dictionary.rev_dict[ele] for ele in batch[-20:]]))
            #print("Batch: ", batch.shape, batch[-20])
            #print("Examples: ", examples.shape, examples[-50:])
            #print("Filtered examples: ", filtered_examples.shape, filtered_examples[-50:])

        timestamp = time.time()
        pearson, spearman, sim_summary = wordsim353.calculate_similarity(session)
        writer.add_summary(sim_summary, newstep)
        if show % 20 == 0:
            log.print("WordSim-353 Pearson: ", "%.3f" % pearson)
            log.print("WordSim-353 Spearman:", "%.3f" % spearman)
        wordsim353_time = time.time() - timestamp
        timestamp = time.time()
        pearson, spearman, sim_summary = simlex999.calculate_similarity(session)
        writer.add_summary(sim_summary, newstep)
        if show % 20 == 0:
            log.print("SimLex-999 Pearson: ", "%.3f" % pearson)
            log.print("SimLex-999 Spearman:", "%.3f" % spearman)
        simlex999_time = time.time() - timestamp
        timestamp = time.time()
        if show % 10 == 0:
            summary, score = analogy.calculate_analogy(session, embeddings.size)
            writer.add_summary(summary, newstep)
            log.print("Analogy score: ", "%.10f" % score)
        analogy_time = time.time() - timestamp
        timeadj = wordsim353_time + simlex999_time + analogy_time
        if timeadj > 10:
            log.print("Evaluation times: ",
                      "%.2f" % wordsim353_time,
                      "%.2f" % simlex999_time,
                      "%.2f" % analogy_time,
                      "("+str(int(timeadj))+"s total)")

        if show % 200 == 0:
            log.print("Waiting for old tasks to complete...")
            alive_tasks = []
            for task in tasks:
                if task.is_alive():
                    alive_tasks.append(task)
            if len(alive_tasks) == 0:
                tasks = []
                timestamp = time.time()
                norm_tensor = embeddings.normalized_tensor().eval()
                log.print("Tensor normalized in", ("%.5f" % (time.time() - timestamp)), "s")
                log.print("Dumping embeddings")
                output_file = args.output + 'embeddings-prov-' + str(embeddings.size) + '-' + get_num_stems_str(args.iterations) + '.tsv'
                dump_task = EmbeddingsDumper(
                    embeddings,
                    norm_tensor,
                    dictionary,
                    save_to=output_file
                )
                dump_task.start()
                analogy_task = AnalogyCalculation(analogy, norm_tensor, log)
                analogy_task.start()
                tasks.append(dump_task)
                tasks.append(analogy_task)
                del norm_tensor
            else:
                log.print("Skipping analogy calculation and embeddings dumping because previous run hasn't completed yet")
                tasks = alive_tasks

    coord.request_stop()
    coord.join(threads)

    del relations
    del text_reader
    timestamp = time.time()
    norm_tensor = embeddings.normalized_tensor().eval()
    log.print("Tensor normalized in", ("%.5f" % (time.time() - timestamp)), "s")

# Step 6: Dump embeddings to file
log.print("Dumping embeddings on disk")
control_tasks(tasks)
dump_task = EmbeddingsDumper(
    embeddings,
    norm_tensor,
    dictionary,
    args.output+'embeddings-' + str(embeddings.size) + '-' + get_num_stems_str(args.iterations) + '.tsv'
)
dump_task.start()
dump_task.join()

#log.print("Waiting for evaluation dumping to finish...")
#control_evaluation(tasks)
log.print("Done")
log.close()