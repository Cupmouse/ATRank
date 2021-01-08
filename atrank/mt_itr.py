import threading
import queue

BUFFER_SIZE = 128

class MultiThreadedIterator:

  def __init__(self, itr):
    self.wrapped = itr
    self.queue = queue.Queue(BUFFER_SIZE)
    self.stop = False
    def loop():
      while not self.stop:
        try:
          n = next(self.wrapped)
        except StopIteration as e:
          while not self.stop:
            try:
              self.queue.put(e, timeout=1)
              break
            except queue.Full:
              continue
          return
        while not self.stop:
          try:
            self.queue.put(n, timeout=1)
            break
          except queue.Full:
            continue
    self.th = threading.Thread(target=loop, daemon=True)
    self.th.start()

  def __iter__(self):
    return self
    
  def __next__(self):
    i = self.queue.get()
    print('itr', end='')
    if isinstance(i, StopIteration):
      raise StopIteration() from i
    return i

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.close()

  def close(self):
    self.stop = True
    self.th.join()

