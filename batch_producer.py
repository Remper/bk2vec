from __future__ import absolute_import
from __future__ import print_function

import struct
import tensorflow as tf

FILENAME = 'test.test'
BATCH_SIZE = 256
WINDOW_SIZE = 3

# Proof of concept application that stores numbers in file and then uses tensorflow to load it back into memory
with open(FILENAME, 'wb') as file:
    for ele in range(1000):
        file.write(struct.pack('=l', ele))

# Loading stream back to memory
graph = tf.Graph()

with graph.as_default():
    filename_queue = tf.train.string_input_producer([FILENAME])
    reader = tf.FixedLengthRecordReader(record_bytes=4 * BATCH_SIZE)
    key, value = reader.read(filename_queue)

    batch = tf.decode_raw(value, tf.int32)
    expanded_batch = tf.expand_dims(batch, 1)
    examples = list()


    def append_example(example):
        examples.append(tf.concat(1, [expanded_batch, tf.expand_dims(example, 1)]))


    for window in range(1, WINDOW_SIZE+1):
        append_example(tf.slice(tf.pad(batch, [[window, 0]]), [0], [BATCH_SIZE]))
        append_example(tf.pad(tf.slice(batch, [window], [BATCH_SIZE - window]), [[0, window]]))
    examples = tf.concat(0, examples)
    examples = tf.train.shuffle_batch([examples],
                                      batch_size=32,
                                      num_threads=8,
                                      capacity=10000,
                                      min_after_dequeue=100,
                                      enqueue_many=True)

with tf.Session(graph=graph) as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            examples_result = sess.run(examples)
            print("  ", examples_result[0], examples_result[len(examples_result)-1])
    except tf.errors.OutOfRangeError:
        print("Epoch concluded")

    coord.request_stop()
    coord.join(threads)
