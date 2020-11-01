import os
from datetime import datetime


class Logger():
    def __init__(self, save_dir, filename='log.txt'):
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, filename)

    def log(self, msg):
        print(msg)
        now = datetime.now()
        str_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        f = open(self.save_path, 'a+')
        w_msg = "[{}] {}\n".format(str_now, msg)
        f.write(w_msg)
        f.close()