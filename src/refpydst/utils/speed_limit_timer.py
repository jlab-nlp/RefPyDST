# time to restrict query speed
import time


class SpeedLimitTimer:
    """
    A timer whicch can be used to meter API requests to OpenAI, etc.
    """
    def __init__(self, second_per_step=0.75):
        self.record_time = time.time()
        self.second_per_step = second_per_step

    def step(self):
        time_div = time.time() - self.record_time
        if time_div <= self.second_per_step:
            time.sleep(self.second_per_step - time_div)
        self.record_time = time.time()

    def sleep(self, s):
        time.sleep(s)


if __name__ == '__main__':
    timer = SpeedLimitTimer()
    start = time.time()
    b = start
    for i in range(0, 50):
        a = b
        timer.step()
        b = time.time()  # current epoch time
        print(i, b-start, b-a)