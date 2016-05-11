from __future__ import absolute_import
from __future__ import print_function

from threading import Lock

print_lock = Lock()


def synchronized(lock):
    """ Synchronization decorator. """

    def wrap(f):
        def newFunction(*args, **kw):
            lock.acquire()
            try:
                return f(*args, **kw)
            finally:
                lock.release()

        return newFunction

    return wrap


def get_num_stems_str(num_steps):
    letter = ''
    divider = 1
    mappings = {
        'b': 1000000000,
        'm': 1000000,
        'k': 1000
    }
    for mapping in mappings:
        if num_steps > mappings[mapping] and mappings[mapping] > divider:
            letter = mapping
            divider = mappings[mapping]
    if num_steps % divider == 0:
        return str(num_steps // divider) + letter
    else:
        return "{:.1f}".format(float(num_steps) / divider) + letter


class Log:
    def __init__(self, args):
        self._log = None
        self._args = args
        self._lock = Lock()

    def init_log(self):
        self._log = open(self._args.output + 'training.log', 'wb')
        self._log.write(str(vars(self._args)))
        self._log.write('\n')

    @synchronized(print_lock)
    def print(self, *args):
        if self._log is None:
            self.init_log()
        print(*args)
        self._log.write(" ".join([str(ele) for ele in args]))
        self._log.write('\n')

    def close(self):
        self._log.close()
        self._log = None