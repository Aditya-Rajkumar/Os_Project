print("LOADING SJF FILE")
import psutil
import os

class SJFScheduler:
    JOB_LENGTHS = {
        'cpu':    'short',
        'memory': 'medium',
        'io':     'long'
    }

    def get_priority(self, task_type):
        length = self.JOB_LENGTHS.get(task_type, 'medium')
        if length == 'short':
            return psutil.ABOVE_NORMAL_PRIORITY_CLASS
        elif length == 'medium':
            return psutil.NORMAL_PRIORITY_CLASS
        else:
            return psutil.BELOW_NORMAL_PRIORITY_CLASS

    def get_priority_name(self, task_type):
        length = self.JOB_LENGTHS.get(task_type, 'medium')
        if length == 'short':
            return 'ABOVE_NORMAL'
        elif length == 'medium':
            return 'NORMAL'
        else:
            return 'BELOW_NORMAL'

    def set_process_priority(self, task_type):
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(self.get_priority(task_type))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass