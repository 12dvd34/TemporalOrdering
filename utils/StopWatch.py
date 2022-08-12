import time


class StopWatch:

    def __init__(self, name="stopwatch"):
        self.name = name
        self.beginning = time.time()

    def restart(self):
        self.beginning = time.time()

    def stop(self):
        end = time.time()
        print(self.name + " done in " + str(round(end - self.beginning, 1)) + "s")