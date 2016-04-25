from __future__ import absolute_import
from __future__ import print_function

import argparse


class Arguments():
    DEFAULT_ITERATIONS = 2000001  # Amount of steps
    DEFAULT_BATCH_SIZE = 128  # Size of the batch for every iteration
    DEFAULT_EMBEDDING_SIZE = 90  # Dimension of the embedding vector.
    DEFAULT_WINDOW_SIZE = 2  # Default size of the context
    DEFAULT_NUM_SKIPS = 2  # How many times to reuse an input to generate a label..
    DEFAULT_NUM_SAMPLED = 64  # Number of negative examples to sample.

    def __init__(self):
        parser = argparse.ArgumentParser(description='Train Word2Vec.')

        parser.add_argument('--iterations', default=Arguments.DEFAULT_ITERATIONS,
                            help='Number of iterations (default {0})'.format(Arguments.DEFAULT_ITERATIONS), metavar='#')
        parser.add_argument('--batch_size', default=Arguments.DEFAULT_BATCH_SIZE,
                            help='Batch size (default {0})'.format(Arguments.DEFAULT_BATCH_SIZE), metavar='#')
        parser.add_argument('--embeddings_size', default=Arguments.DEFAULT_EMBEDDING_SIZE,
                            help='Embeddings size (default {0})'.format(Arguments.DEFAULT_EMBEDDING_SIZE), metavar='#')
        parser.add_argument('--window_size', default=Arguments.DEFAULT_WINDOW_SIZE,
                            help='Window size (default {0})'.format(Arguments.DEFAULT_WINDOW_SIZE), metavar='#')
        parser.add_argument('--num_skips', default=Arguments.DEFAULT_NUM_SKIPS,
                            help='Num skips (default {0})'.format(Arguments.DEFAULT_NUM_SKIPS), metavar='#')
        parser.add_argument('--num_sampled', default=Arguments.DEFAULT_NUM_SAMPLED,
                            help='Num sampled (default {0})'.format(Arguments.DEFAULT_NUM_SAMPLED), metavar='#')
        parser.add_argument('--output', default='', help='Output dir (default - current dir)', metavar='#')
        parser.add_argument('--clean', default=False, action='store_true',
                            help='Calculate only plain skipgram objective')
        parser.add_argument('--detached', default=False, action='store_true',
                            help="Calculate category objective independently of the main one (select it's own samples)")
        parser.add_argument('--no-margin', default=False, action='store_true',
                            help="Disable margin from category objective")
        self.args = parser.parse_args()
        self.args.batch_size = int(self.args.batch_size)
        self.args.iterations = int(self.args.iterations)
        self.args.embeddings_size = int(self.args.embeddings_size)
        self.args.num_skips = int(self.args.num_skips)
        self.args.num_sampled = int(self.args.num_sampled)
        self.args.window_size = int(self.args.window_size)
        self.args.clean = bool(self.args.clean)
        self.args.detached = bool(self.args.detached)
        self.args.no_margin = bool(self.args.no_margin)
        if self.args.output != '' and not self.args.output.endswith('/'):
            self.args.output += '/'

    def show_args(self):
        print("Initialized with settings:")
        print(vars(self.args))
        return self
