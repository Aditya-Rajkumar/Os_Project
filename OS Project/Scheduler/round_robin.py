import psutil
import os
import time
import threading

class RoundRobinScheduler:
    QUANTUM = 0.1

    def __init__(self):
        self.task_types = ['cpu', 'memory', 'io']
        self.current_index = 0
        self.last_switch = time.time()
        self.lock = threading.Lock()

    def get_current_turn(self):
        with self.lock:
            now = time.time()
            if now - self.last_switch >= self.QUANTUM:
                self.current_index = (self.current_index + 1) % len(self.task_types)
                self.last_switch = now
            return self.task_types[self.current_index]

    def get_priority(self, task_type):
        current_turn = self.get_current_turn()
        if task_type == current_turn:
            return psutil.ABOVE_NORMAL_PRIORITY_CLASS
        else:
            return psutil.BELOW_NORMAL_PRIORITY_CLASS

    def get_priority_name(self, task_type):
        current_turn = self.get_current_turn()
        if task_type == current_turn:
            return 'ABOVE_NORMAL'
        else:
            return 'BELOW_NORMAL'

    def set_process_priority(self, task_type):
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(self.get_priority(task_type))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass