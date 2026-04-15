from Scheduler.fcfs import FCFSScheduler
from Scheduler.sjf import SJFScheduler
from Scheduler.round_robin import RoundRobinScheduler
from Scheduler.ddqn import DDQNScheduler, ddqn_lock, _ddqn_instance

schedulers = {
    'FCFS': FCFSScheduler(),
    'SJF':  SJFScheduler(),
    'RR':   RoundRobinScheduler(),
    'DDQN': _ddqn_instance    # same shared instance from ddqn.py
}

def set_priority(scheduler_name, task_type):
    scheduler = schedulers.get(scheduler_name.upper())
    if scheduler:
        scheduler.set_process_priority(task_type)

def get_priority_name(scheduler_name, task_type):
    scheduler = schedulers.get(scheduler_name.upper())
    if scheduler:
        return scheduler.get_priority_name(task_type)
    return 'NORMAL'