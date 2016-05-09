from __future__ import absolute_import
from __future__ import print_function

from threading import Thread

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

import gzip
import csv
import numpy as np
import tensorflow as tf


class WordSimilarity:
    def __init__(self, name, pairs, labels):
        self._pairs = np.array(pairs)
        self._labels = np.array(labels)
        self._op = None
        self._input_op = None
        self._name = name

    def get_pairs(self):
        return self._pairs

    def produce_summary(self, words1, words2, metric, summary=None):
        distances = np.diag(cdist(words1, words2, metric))
        pearson = pearsonr(distances, self._labels)
        spearman = spearmanr(distances, self._labels)
        if summary is None:
            summary = tf.Summary()
        value = summary.value.add()
        value.tag = self._name+"/"+metric+"/pearson"
        value.simple_value = pearson[0]
        value = summary.value.add()
        value.tag = self._name+"/"+metric+"/spearman"
        value.simple_value = spearman.correlation
        return summary, pearson[0], spearman.correlation

    def calculate_similarity(self, session):
        if self._op is None:
            print("Operation hasn't been registered. Ignoring")
            return 0, 0

        vectors = self._op.eval(session=session)
        words1 = vectors[:, 0, :]
        words2 = vectors[:, 1, :]
        summary, _, _ = self.produce_summary(words1, words2, 'euclidean')
        summary, pearson, spearman = self.produce_summary(words1, words2, 'cosine', summary)
        return pearson, spearman, summary

    def register_op(self, embeddings):
        if self._op is None:
            with tf.name_scope("similarity_calculation"):
                self._input_op = tf.constant(self._pairs, name="similarity_pairs")
                self._op = tf.gather(embeddings, self._input_op)
        return self._op

    @staticmethod
    def wordsim353(source, dictionary):
        pairs = list()
        labels = list()
        with open(source, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, quotechar='')
            count = 0
            for row in reader:
                count += 1
                if count == 1:
                    print("Skipping header")
                    continue
                if len(row) != 3:
                    print("Inconsistent file")
                    continue
                if row[0] not in dictionary:
                    print("Word", row[0], "not in dictionary")
                    continue
                if row[1] not in dictionary:
                    print("Word", row[1], "not in dictionary")
                    continue
                pairs.append([dictionary[row[0]], dictionary[row[1]]])
                labels.append(10 - float(row[2]))
        return WordSimilarity("wordsim353", pairs, labels)

    @staticmethod
    def simlex999(source, dictionary):
        pairs = list()
        labels = list()
        with open(source, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
            count = 0
            for row in reader:
                count += 1
                if count == 1:
                    print("Skipping header")
                    continue
                if len(row) != 10:
                    print("Inconsistent file")
                    continue
                if row[0] not in dictionary:
                    print("Word", row[0], "not in dictionary")
                    continue
                if row[1] not in dictionary:
                    print("Word", row[1], "not in dictionary")
                    continue
                pairs.append([dictionary[row[0]], dictionary[row[1]]])
                value = 10 - float(row[3])
                labels.append(value)
        return WordSimilarity("simlex999", pairs, labels)


class EvaluationDumper(Thread):
    def __init__(self, evaluation, postfix, folder=''):
        Thread.__init__(self)
        self._evaluation = evaluation
        self.postfix = postfix
        self._folder = folder

    def run(self):
        count = 0
        with gzip.open(self._folder+'categories-' + self.postfix + '.tsv.gz', 'wb') as writer:
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


def dump_evaluation(evaluation, postfix, folder=''):
    dumper = EvaluationDumper(evaluation, postfix, folder=folder)
    dumper.start()
    return dumper


def control_evaluation(threads):
    for dumper in threads:
        dumper.join()