from __future__ import absolute_import
from __future__ import print_function

from threading import Thread
import gzip


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