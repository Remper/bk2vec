from __future__ import absolute_import
from __future__ import print_function

import gzip
import struct
import time
import tensorflow as tf
from bk2vec.dictionary import Dictionary

DEFAULT_BATCH_SIZE = 1024
DEFAULT_THREADS = 2
DEFAULT_READERS = 2
DICTIONARY_THRESHOLD = 4


class TextReader:
    def __init__(self, filename, window_size, threads=DEFAULT_THREADS, batch_size=DEFAULT_BATCH_SIZE):
        print("Text reader initialized with threads (" + str(threads) + "), batch size (" + str(batch_size) + ")")
        self.filename = filename
        self._window_size = window_size
        self._reading_op = None
        self._category_op = None
        self._relations_op = None
        self._debug_op = None
        self._threads = threads
        self._batch_size = batch_size
        self._reader_stats = None
        self._relations = None
        self._add_categories = False
        self._add_relations = False

    def set_relations(self, relations):
        self._relations = relations

    def enable_categories(self):
        if self._relations is not None:
            self._add_categories = True

    def enable_relations(self):
        if self._relations is not None:
            self._add_relations = True
            self.enable_categories()

    def get_category_ops(self):
        if self._category_op is None:
            raise Exception("Category operation wasn't initialized")
        return self._category_op

    def get_relations_ops(self):
        if self._relations_op is None:
            raise Exception("Relations operation wasn't initialized")
        return self._relations_op

    def get_reading_ops(self):
        if self._reading_op is None:
            filename_queue = tf.train.string_input_producer([self.filename])
            reader = tf.FixedLengthRecordReader(record_bytes=4 * DEFAULT_BATCH_SIZE)
            self._reader_stats = reader.num_records_produced()
            key, value = reader.read(filename_queue)

            # Converting bytes into int32s and stating that this is a scalar
            words = tf.reshape(tf.decode_raw(value, tf.int32), [-1])
            batch = tf.train.batch([words],
                                   batch_size=self._batch_size,
                                   num_threads=DEFAULT_READERS,
                                   enqueue_many=True,
                                   capacity=self._batch_size * 10,
                                   name="word_buffer")
            expanded_batch = tf.expand_dims(batch, 1)
            examples = list()

            def append_example(example, name):
                examples.append(tf.concat(1, [expanded_batch, tf.expand_dims(example, 1)], name='batch-window' + name))

            for window in range(1, self._window_size + 1):
                append_example(tf.slice(tf.pad(batch, [[window, 0]]), [0], [self._batch_size]), '-p' + str(window))
                append_example(tf.pad(tf.slice(batch, [window], [self._batch_size - window]), [[0, window]]),
                               '-m' + str(window))
            examples = tf.concat(0, examples)

            # Filtering zeros
            not_zeros, _ = tf.unique(tf.squeeze(tf.slice(tf.where(tf.not_equal(examples[:, 1], 0)), [0, 0], [-1, 1])))
            filtered_examples = tf.gather(examples, not_zeros)
            self._debug_op = [batch, examples, filtered_examples]

            examples = tf.train.shuffle_batch([filtered_examples],
                                              batch_size=self._batch_size,
                                              num_threads=self._threads,
                                              capacity=self._batch_size * self._threads * 5,
                                              min_after_dequeue=self._batch_size * self._threads,
                                              enqueue_many=True,
                                              name="batch_buffer")

            ops = [examples]
            if self._add_categories:
                categorical_examples, categories_populated = tf.py_func(
                    lambda examples: self._relations.generate_categorical_batch(examples),
                    [examples], [tf.int32, tf.int32],
                    name="categorical_batch"
                )
                ops.append(tf.reshape(categorical_examples, [-1, 2]))

                if self._add_relations:
                    relational_examples = tf.py_func(
                        lambda examples, categories: self._relations.generate_relational_batch(examples, categories),
                        [examples, categories_populated], [tf.int32],
                        name="relational_batch"
                    )[0]
                    ops.append(tf.reshape(relational_examples, [-1, 5]))

            batch_queue_capacity = self._threads * len(ops) * 10
            batch_queue = tf.FIFOQueue(
                capacity=batch_queue_capacity,
                dtypes=[tf.int32] * len(ops),
                name="preprocessing_buffer"
            )
            tf.scalar_summary(
                "queue/%s/fraction_of_%d_full" % (batch_queue.name, batch_queue_capacity),
                tf.cast(batch_queue.size(), tf.float32) * (1. / batch_queue_capacity)
            )
            queue_runner = tf.train.QueueRunner(batch_queue, [batch_queue.enqueue(ops)] * self._threads)
            tf.train.add_queue_runner(queue_runner)
            example_tuple = batch_queue.dequeue()

            if not isinstance(example_tuple, list):
                reading_op = example_tuple
            else:
                reading_op = example_tuple[0]
            # TODO: don't split if possible
            self._reading_op = tf.split(1, 2, reading_op)
            if self._add_categories:
                self._category_op = example_tuple[1]
            if self._add_relations:
                self._debug_op.append(example_tuple[2])
                self._relations_op = example_tuple[2]
        return self._reading_op

    def get_reader_stats(self, session):
        if self._reader_stats is None:
            raise Exception("Reader stats operation wasn't initialized")
        return self._reader_stats.eval(session=session) * DEFAULT_BATCH_SIZE

    def get_debug_op(self):
        if self._debug_op is None:
            raise Exception("Debug operation wasn't initialized")
        return self._debug_op

    def text2bin(self, source, blacklist=False, threshold=None):
        dictionary, counts = TextReader.build_dictionary(source, blacklist_enabled=blacklist, threshold=threshold)
        print("Dumping counts")
        timestamp = time.time()
        with open(self.filename + '_counts.tsv', 'wb') as file:
            for ele in counts.keys():
                file.write(ele)
                file.write('\t')
                file.write(str(counts[ele]))
                file.write('\n')
        print("Done in", ("%.3f" % (time.time() - timestamp)) + "s")
        del counts
        self.store_dictionary(dictionary)
        print("Condensing text stream")
        processed = 0
        timestamp = time.time()
        with open(self.filename, 'wb') as file:
            for word in TextReader.words(source):
                if word not in dictionary.dict:
                    continue
                idx = dictionary.dict[word]
                file.write(struct.pack('=l', idx))
                processed += 1
                if processed % 100000000 == 0:
                    print("  ", str(processed // 1000000) + "m words parsed,",
                          "time:", ("%.3f" % (time.time() - timestamp)) + "s")
                    timestamp = time.time()
            if processed % DEFAULT_BATCH_SIZE != 0:
                print("  Padding stream for ", DEFAULT_BATCH_SIZE - processed % DEFAULT_BATCH_SIZE, "words")
                while processed % DEFAULT_BATCH_SIZE != 0:
                    file.write(struct.pack('=l', 0))
                    processed += 1
        return dictionary

    def store_dictionary(self, dictionary):
        print("Dumping dictionary")
        timestamp = time.time()
        with open(self.filename + '_dict.tsv', 'wb') as file:
            for ele in dictionary.dict.keys():
                file.write(ele)
                file.write('\t')
                file.write(str(dictionary.dict[ele]))
                file.write('\n')
        print("Done in", ("%.3f" % (time.time() - timestamp)) + "s")

    def restore_dictionary(self):
        print('Restoring dictionary')
        timestamp = time.time()
        processed = 0
        dictionary = Dictionary()
        with open(self.filename + "_dict.tsv", 'rb') as file:
            for row in file:
                row = row.split('\t')
                dictionary.put_word(int(row[1]), row[0])
                processed += 1
                if processed % 3000000 == 0:
                    print("  " + str(processed // 1000000) + "m words parsed")
        print("Done in", ("%.3f" % (time.time() - timestamp)) + "s")
        return dictionary

    @staticmethod
    def words(source):
        try:
            if source.endswith('.gz'):
                file = gzip.open(source, 'rb')
            else:
                file = open(source, 'rb')
        except:
            file = open(source, 'rb')

        for row in file:
            for word in row.split():
                yield word
        file.close()

    @staticmethod
    def build_dictionary(source, blacklist_enabled=False, threshold=None):
        print("Loading blacklist")
        blacklist = list()
        with open('datasets/blacklist.txt') as file:
            for row in file:
                blacklist.append(row.strip())
        print("Blacklist contains", len(blacklist), "words")
        print("Building word frequency list")
        processed = 0
        counts = dict()
        timestamp = time.time()
        for word in TextReader.words(source):
            word = str(word)
            processed += 1
            if processed % 100000000 == 0:
                print("  ", str(processed // 1000000) + "m words parsed (last:", word, ", dict size:", len(counts),
                      ", time:", ("%.3f" % (time.time() - timestamp)) + "s")
                timestamp = time.time()
            if word in counts:
                counts[word] += 1
                continue
            counts[word] = 1
        print("  Parsing finished")
        if threshold is None:
            threshold = DICTIONARY_THRESHOLD
        print("  Removing a tail and assembling a dictionary (Threshold:", threshold, ")")
        filtered_dictionary = Dictionary()
        processed = 0
        timestamp = time.time()
        for word in counts.keys():
            processed += 1
            if processed % 1000000 == 0:
                print("  ", str(processed // 1000000) + "m words parsed,",
                      "time:", ("%.3f" % (time.time() - timestamp)) + "s")
                timestamp = time.time()
            if blacklist_enabled and word in blacklist:
                continue
            if counts[word] >= DICTIONARY_THRESHOLD:
                filtered_dictionary.add_word(word)
        return filtered_dictionary, counts
