from multiprocessing import Semaphore, Value

class Barrier:
    def __init__(self, n):
        self.n       = n
        self.count   = Value('i', 0)
        self.mutex   = Semaphore(1)
        self.barrier = Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count.value += 1
        self.mutex.release()

        if self.count.value == self.n:
            self.barrier.release()

        self.barrier.acquire()
        self.barrier.release()