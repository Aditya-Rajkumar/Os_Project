import sys
sys.path.append('C:/Users/adity/OneDrive/Desktop/OS Project')

from flask import Flask, jsonify
from Scheduler.scheduler_manager import set_priority, get_priority_name
from Scheduler.ddqn import ddqn_lock, _ddqn_instance
import numpy as np
import os
import time
import psutil
import threading
from datetime import datetime

app = Flask(__name__)

task_queue = []
queue_lock = threading.Lock()
current_scheduler = 'FCFS'
metrics = []

def record_metric(task_type, arrival_time, completion_time,
                  response_time, scheduler, action):
    metrics.append({
        'timestamp':        datetime.now().strftime('%H:%M:%S.%f'),
        'task_type':        task_type,
        'pid':              os.getpid(),
        'arrival_time':     arrival_time,
        'completion_time':  completion_time,
        'response_time':    response_time,
        'scheduler_used':   scheduler,
        'action_taken':     action,
        'system_cpu_load':  psutil.cpu_percent(interval=0.1),
        'memory_available': psutil.virtual_memory().available / (1024**3)
    })

@app.route('/cpu-task')
def cpu_task():
    arrival_time = time.time()
    start = time.time()
    matrix = np.random.rand(200, 200)
    result = np.linalg.eig(matrix)
    end = time.time()
    completion_time = (end - start) * 1000
    response_time = (end - arrival_time) * 1000
    if current_scheduler == 'DDQN':
        with ddqn_lock:
            _ddqn_instance.set_process_priority('cpu', completion_time=completion_time, response_time=response_time)
    else:
        set_priority(current_scheduler, 'cpu')
    record_metric('cpu', arrival_time, completion_time, response_time, current_scheduler, get_priority_name(current_scheduler, 'cpu'))
    return jsonify({'task_type': 'cpu', 'completion_time': completion_time, 'response_time': response_time, 'scheduler': current_scheduler})

@app.route('/memory-task')
def memory_task():
    arrival_time = time.time()
    start = time.time()
    data = np.random.rand(10000, 1000)
    result = np.sum(data)
    end = time.time()
    completion_time = (end - start) * 1000
    response_time = (end - arrival_time) * 1000
    if current_scheduler == 'DDQN':
        with ddqn_lock:
            _ddqn_instance.set_process_priority('memory', completion_time=completion_time, response_time=response_time)
    else:
        set_priority(current_scheduler, 'memory')
    record_metric('memory', arrival_time, completion_time, response_time, current_scheduler, get_priority_name(current_scheduler, 'memory'))
    return jsonify({'task_type': 'memory', 'completion_time': completion_time, 'response_time': response_time, 'scheduler': current_scheduler})

@app.route('/io-task')
def io_task():
    arrival_time = time.time()
    unique_filename = f'temp_io_{os.getpid()}_{time.time_ns()}.txt'
    start = time.time()
    try:
        with open(unique_filename, 'w') as f:
            f.write('x' * 10000000)
        with open(unique_filename, 'r') as f:
            data = f.read()
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
    end = time.time()
    completion_time = (end - start) * 1000
    response_time = (end - arrival_time) * 1000
    if current_scheduler == 'DDQN':
        with ddqn_lock:
            _ddqn_instance.set_process_priority('io', completion_time=completion_time, response_time=response_time)
    else:
        set_priority(current_scheduler, 'io')
    record_metric('io', arrival_time, completion_time, response_time, current_scheduler, get_priority_name(current_scheduler, 'io'))
    return jsonify({'task_type': 'io', 'completion_time': completion_time, 'response_time': response_time, 'scheduler': current_scheduler})

@app.route('/set-scheduler/<name>')
def set_scheduler(name):
    global current_scheduler
    current_scheduler = name.upper()
    return jsonify({'scheduler': current_scheduler})

@app.route('/get-metrics')
def get_metrics():
    import pandas as pd
    if metrics:
        df = pd.DataFrame(metrics)
        summary = {
            'scheduler':           current_scheduler,
            'total_requests':      len(metrics),
            'avg_completion_time': df['completion_time'].mean(),
            'avg_response_time':   df['response_time'].mean(),
            'throughput':          len(metrics) / df['completion_time'].sum() * 1000
        }
        return jsonify(summary)
    return jsonify({'message': 'no metrics yet'})

@app.route('/save-metrics/<load>')
def save_metrics(load):
    import pandas as pd
    if metrics:
        df = pd.DataFrame(metrics)
        path = f'C:/Users/adity/OneDrive/Desktop/OS Project/dataset/{current_scheduler}_{load}_metrics.csv'
        df.to_csv(path, index=False)
        metrics.clear()
        return jsonify({'saved': path})
    return jsonify({'message': 'no metrics to save'})

if __name__ == '__main__':
    app.run(port=5000, threaded=True)