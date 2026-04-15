print("LOADING FCFS FILE")
try:
    import psutil
except Exception as e:
    print("ERROR:", e)
import psutil
import os

class FCFSScheduler:
    def get_priority(self, task_type):
        return psutil.NORMAL_PRIORITY_CLASS

    def get_priority_name(self, task_type):
        return 'NORMAL'

    def set_process_priority(self, task_type):
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(self.get_priority(task_type))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass