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