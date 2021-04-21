import sys


class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'a')
    
    def write(self, message):
        self.terminal.write(messagge)
        self.log.write(message)
    
    def close(self):
        sys.stdout = sys.__stdout__
        self.log.close()
    
    def flush(self):
        pass