from __future__ import absolute_import
from __future__ import print_function

import argparse
import random

class AbstractArguments():
    def __init__(self, parser):
        self.args = parser.parse_args()

    def show_args(self):
        print("Initialized with settings:")
        print(vars(self.args))
        return self


class Arguments(AbstractArguments):
    DEFAULT_ITERATIONS = 5000001  # Amount of steps
    DEFAULT_BATCH_SIZE = 256  # Size of the batch for every iteration
    DEFAULT_EMBEDDING_SIZE = 128  # Dimension of the embedding vector.
    DEFAULT_WINDOW_SIZE = 3  # Default size of the context
    DEFAULT_NUM_SKIPS = 2  # How many times to reuse an input to generate a label..
    DEFAULT_NUM_SAMPLED = 64  # Number of negative examples to sample.
    DEFAULT_MARGIN = 1.0 # Default margin for the category objective
    DEFAULT_NUM_THREADS = 8 # Default amount of threads for prefetching/processing
    DEFAULT_MODE = 'attached'

    def __init__(self):
        AbstractArguments.__init__(self, self.get_parser())
        self.args.batch_size = int(self.args.batch_size)
        self.args.iterations = int(self.args.iterations)
        self.args.embeddings_size = int(self.args.embeddings_size)
        self.args.num_skips = int(self.args.num_skips)
        self.args.num_sampled = int(self.args.num_sampled)
        self.args.window_size = int(self.args.window_size)
        self.args.threads = int(self.args.threads)
        self.args.seed = int(self.args.seed)
        self.args.margin = float(self.args.margin)
        self.args.clean = bool(self.args.clean)
        self.args.notoken = bool(self.args.notoken)
        if self.args.output != '' and not self.args.output.endswith('/'):
            self.args.output += '/'

    def get_parser(self):
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
        parser.add_argument('--threads', default=Arguments.DEFAULT_NUM_THREADS,
                            help='Num threads (default {0})'.format(Arguments.DEFAULT_NUM_THREADS), metavar='#')
        parser.add_argument('--num_sampled', default=Arguments.DEFAULT_NUM_SAMPLED,
                            help='Num sampled (default {0})'.format(Arguments.DEFAULT_NUM_SAMPLED), metavar='#')
        parser.add_argument('--seed', default=random.randint(0, 2147483647), help='Seed (default random)', metavar='#')
        parser.add_argument('--margin', default=Arguments.DEFAULT_MARGIN, metavar='#',
                            help='Margin for category objective (default {0})'.format(Arguments.DEFAULT_MARGIN))
        parser.add_argument('--mode', default=Arguments.DEFAULT_MODE, metavar='#',
                            help='Category objective type ("attached", "detached", "combo", default "{0}")'.format(Arguments.DEFAULT_MODE))
        parser.add_argument('--notoken', default=False, action='store_true',
                            help='User untokenized version of categories')
        parser.add_argument('--output', default='', help='Output dir (default - current dir)', metavar='#')
        parser.add_argument('--clean', default=False, action='store_true',
                            help='Calculate only plain skipgram objective')
        return parser


class EvaluationArguments(AbstractArguments):
    def __init__(self):
        AbstractArguments.__init__(self, self.get_parser())

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Evaluate Word2Vec.')

        parser.add_argument('--test', default='', required=True, help='Location of test set', metavar='#')
        parser.add_argument('--embeddings', required=True, help='Location of embeddings', metavar='#')
        return parser


class FilterArguments(AbstractArguments):
    def __init__(self):
        AbstractArguments.__init__(self, self.get_parser())
        if self.args.output != '' and not self.args.output.endswith('/'):
            self.args.output += '/'

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Evaluate Word2Vec.')

        parser.add_argument('--test', default='', required=True, help='Location of test set', metavar='#')
        parser.add_argument('--output', required=True, help='Output folder', metavar='#')
        return parser


class RenderArguments(AbstractArguments):
    def __init__(self):
        AbstractArguments.__init__(self, self.get_parser())

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Evaluate Word2Vec.')

        parser.add_argument('--indices', default='', required=True, help='Location of indices to render', metavar='#')
        parser.add_argument('--embeddings', required=True, help='Location of embeddings', metavar='#')
        return parser
