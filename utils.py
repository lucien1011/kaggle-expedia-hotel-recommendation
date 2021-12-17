import time
from tqdm import tqdm

class Timer(object):
    def __init__(self,mode='tqdm'):
        self._start_time = None
        self._timestamps = []
        self._mode = mode

    def start(self):
        self._start_time = time.time()

    def checkpoint(self):
        self._timestamps.append(time.time())

    def check(self):
        secs = int(time.time() - self._start_time)
        if self._mode == 'tqdm':
            tqdm.write(f'\ttime: {secs}[s]')
        elif self._mode == 'python':
            print(f'\ttime: {secs}[s]')

    def reset(self):
        self._start_time = None
        self._timestamps = []

    def print_reset(self):
        assert bool(self._start_time)
        self.check()
        self.reset()
