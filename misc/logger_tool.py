import sys
import time


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

