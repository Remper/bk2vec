from __future__ import absolute_import
from __future__ import print_function

import argparse


class Arguments():
  DEFAULT_ITERATIONS = 2000001  # Amount of steps
  DEFAULT_BATCH_SIZE = 128      # Size of the batch for every iteration
  DEFAULT_EMBEDDING_SIZE = 90   # Dimension of the embedding vector.
  DEFAULT_WINDOW_SIZE = 2       # Default size of the context
  DEFAULT_NUM_SKIPS = 2         # How many times to reuse an input to generate a label..
  DEFAULT_NUM_SAMPLED = 64      # Number of negative examples to sample.

  def __init__(self):
    parser = argparse.ArgumentParser(description='Train Word2Vec.')

    parser.add_argument('--iterations', default=Arguments.DEFAULT_ITERATIONS,
                        help='Number of iterations (default {0})'.format(Arguments.DEFAULT_ITERATIONS), metavar='#')
    parser.add_argument('--batch_size', default=Arguments.DEFAULT_BATCH_SIZE,
                        help='Batch size (default {0})'.format(Arguments.DEFAULT_BATCH_SIZE), metavar='#')
    parser.add_argument('--embedding_size', default=Arguments.DEFAULT_EMBEDDING_SIZE,
                    help='Embedding size (default {0})'.format(Arguments.DEFAULT_EMBEDDING_SIZE), metavar='#')
    parser.add_argument('--window_size', default=Arguments.DEFAULT_WINDOW_SIZE,
                        help='Window size (default {0})'.format(Arguments.DEFAULT_WINDOW_SIZE), metavar='#')
    parser.add_argument('--num_skips', default=Arguments.DEFAULT_NUM_SKIPS,
                        help='Num skips (default {0})'.format(Arguments.DEFAULT_NUM_SKIPS), metavar='#')
    parser.add_argument('--num_sampled', default=Arguments.DEFAULT_NUM_SAMPLED,
                        help='Num sampled (default {0})'.format(Arguments.DEFAULT_NUM_SAMPLED), metavar='#')
    self.args = parser.parse_args()

  def show_args(self):
    print("Initialized with settings:")
    print(vars(self.args))
    return self
