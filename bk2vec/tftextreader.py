from __future__ import absolute_import
from __future__ import print_function

import gzip
import struct
import time
import tensorflow as tf
from bk2vec.dictionary import Dictionary

DEFAULT_BATCH_SIZE = 256
DICTIONARY_THRESHOLD = 2


class TextReader():
    def __init__(self, filename, window_size, threads=8, batch_size=DEFAULT_BATCH_SIZE):
        self.filename = filename
        self._window_size = window_size
        self._reading_op = None
        self._threads = threads
        self._batch_size = batch_size

    def get_reading_ops(self):
        if self._reading_op is None:
            filename_queue = tf.train.string_input_producer([self.filename])
            reader = tf.FixedLengthRecordReader(record_bytes=4 * self._batch_size)
            key, value = reader.read(filename_queue)

            batch = tf.decode_raw(value, tf.int32)
            expanded_batch = tf.expand_dims(batch, 1)
            examples = list()

            def append_example(example, name):
                examples.append(tf.concat(1, [expanded_batch, tf.expand_dims(example, 1)], name='batch-window'+name))

            for window in range(1, self._window_size + 1):
                append_example(tf.slice(tf.pad(batch, [[window, 0]]), [0], [self._batch_size]), '-p'+str(window))
                append_example(tf.pad(tf.slice(batch, [window], [self._batch_size - window]), [[0, window]]), '-m'+str(window))
            examples = tf.concat(0, examples)
            examples = tf.train.shuffle_batch([examples],
                                              batch_size=DEFAULT_BATCH_SIZE,
                                              num_threads=self._threads,
                                              capacity=DEFAULT_BATCH_SIZE*self._threads*3,
                                              min_after_dequeue=DEFAULT_BATCH_SIZE*self._threads,
                                              enqueue_many=True)
            examples = tf.split(1, 2, examples)
            self._reading_op = examples
        return self._reading_op

    def text2bin(self, source, threshold=None):
        dictionary, counts = TextReader.build_dictionary(source, threshold)
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
                file.write(struct.pack('=l', dictionary[word]))
                processed += 1
                if processed % 100000000 == 0:
                    print("  ", str(processed // 1000000) + "m words parsed",
                          ", time:", ("%.3f" % (time.time() - timestamp)) + "s)")
                    timestamp = time.time()
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
    def build_dictionary(source, threshold=None):
        print("Building word frequency list")
        processed = 0
        counts = dict()
        timestamp = time.time()
        for word in TextReader.words(source):
            word = str(word)
            processed += 1
            if processed % 100000000 == 0:
                print("  ", str(processed // 1000000) + "m words parsed (last:", word, ", dict size:", len(counts),
                      ", time:", ("%.3f" % (time.time() - timestamp)) + "s)")
                timestamp = time.time()
            if word in counts:
                counts[word] += 1
                continue
            counts[word] = 1
        print("Parsing finished")
        if threshold is None:
            threshold = DICTIONARY_THRESHOLD
        print("Removing a tail and assembling a dictionary (Threshold: ", threshold, ")")
        filtered_dictionary = Dictionary()
        processed = 0
        timestamp = time.time()
        for word in counts.keys():
            processed += 1
            if processed % 1000000 == 0:
                print("  ", str(processed // 1000000) + "m words parsed,",
                      "(" + ("%.3f" % (time.time() - timestamp)) + "s)")
                timestamp = time.time()
            if counts[word] >= DICTIONARY_THRESHOLD:
                filtered_dictionary.add_word(word)
        return filtered_dictionary, counts
