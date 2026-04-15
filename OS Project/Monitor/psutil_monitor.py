import psutil
import time
import pandas as pd
import os
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        # dataset storage
        self.records = []
        
        # dataset columns matching paper's metrics
        self.columns = [
            'timestamp',
            'task_type',
            'pid',
            'cpu_usage',
            'memory_usage',
            'queue_length',
            'arrival_time',
            'completion_time',
            'response_time',
            'scheduler_used',
            'action_taken',
            'system_cpu_load',
            'system_memory_available'
        ]
        
        # track active nginx worker processes
        self.active_tasks = {}
        
        # dataset save location
        self.save_path = 'C:/Users/adity/OneDrive/Desktop/OS Project/dataset/recordings.csv'

    def get_nginx_workers(self):
        """Get all real Nginx worker processes currently running"""
        nginx_workers = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 
                                          'memory_percent', 'status']):
            try:
                if 'nginx' in proc.info['name'].lower():
                    nginx_workers.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return nginx_workers

    def get_system_state(self):
        """
        Get real system state - this is s_t = (q_t, l_t, r_t) from the paper
        but reading from real Windows OS instead of simulation
        """
        workers = self.get_nginx_workers()
        
        state = {
            # q_t: task queue length (real number of active workers)
            'queue_length': len(workers),
            
            # l_t: real system CPU load from Windows
            'system_cpu_load': psutil.cpu_percent(interval=0.1),
            
            # r_t: real available memory from Windows
            'system_memory_available': psutil.virtual_memory().available / (1024**3),
            
            # extra real metrics beyond paper
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
        return state

    def record_task(self, task_type, pid, arrival_time, 
                    completion_time, response_time,
                    scheduler_used, action_taken):
        """Record one real task measurement into dataset"""
        
        # get real system state at this moment
        state = self.get_system_state()
        
        # get real process metrics
        try:
            proc = psutil.Process(pid)
            cpu_usage = proc.cpu_percent(interval=0.1)
            memory_usage = proc.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cpu_usage = 0
            memory_usage = 0

        # one real row of data
        record = {
            'timestamp':                datetime.now().strftime('%H:%M:%S.%f'),
            'task_type':                task_type,
            'pid':                      pid,
            'cpu_usage':                cpu_usage,
            'memory_usage':             memory_usage,
            'queue_length':             state['queue_length'],
            'arrival_time':             arrival_time,
            'completion_time':          completion_time,
            'response_time':            response_time,
            'scheduler_used':           scheduler_used,
            'action_taken':             action_taken,
            'system_cpu_load':          state['system_cpu_load'],
            'system_memory_available':  state['system_memory_available']
        }
        
        self.records.append(record)
        return record

    def calculate_reward(self, record):
        """
        Paper's reward function extended with fairness and energy
        r = α₁·U − α₂·T + α₃·R + α₄·Fairness − α₅·Energy
        all values from real psutil measurements
        """
        # weighting coefficients
        a1 = 0.3   # resource utilization weight
        a2 = 0.3   # completion time weight
        a3 = 0.2   # response time weight
        a4 = 0.1   # fairness weight
        a5 = 0.1   # energy weight

        # normalize real values
        U = 1 - (record['system_cpu_load'] / 100)
        T = min(record['completion_time'] / 1000, 1)
        R = min(record['response_time'] / 1000, 1)
        
        # fairness: penalize large variance in completion times
        if len(self.records) > 1:
            completion_times = [r['completion_time'] for r in self.records[-10:]]
            fairness = 1 - (max(completion_times) - min(completion_times)) / 1000
            fairness = max(0, min(1, fairness))
        else:
            fairness = 1.0
        
        # energy: real CPU frequency x utilization from psutil
        state = self.get_system_state()
        energy = (state['cpu_freq'] * record['system_cpu_load']) / 100000
        energy = min(energy, 1)

        # final reward from real measurements
        reward = (a1 * U) - (a2 * T) + (a3 * (1-R)) + (a4 * fairness) - (a5 * energy)
        return reward

    def get_summary_metrics(self, scheduler_name):
        """
        Calculate the 3 metrics from paper's Table 1
        from real measurements
        """
        scheduler_records = [r for r in self.records 
                             if r['scheduler_used'] == scheduler_name]
        
        if not scheduler_records:
            return None
        
        df = pd.DataFrame(scheduler_records)
        
        # paper's 3 metrics - from real measurements
        avg_completion_time = df['completion_time'].mean()
        throughput = len(scheduler_records) / df['completion_time'].sum() * 1000
        avg_response_time = df['response_time'].mean()
        
        print(f"\n--- {scheduler_name} Results ---")
        print(f"Average Task Completion Time: {avg_completion_time:.2f} ms")
        print(f"System Throughput:            {throughput:.2f} tasks/sec")
        print(f"Average Response Time:        {avg_response_time:.2f} ms")
        
        return {
            'scheduler': scheduler_name,
            'avg_completion_time': avg_completion_time,
            'throughput': throughput,
            'avg_response_time': avg_response_time
        }

    def save_dataset(self):
        """Save all real measurements to CSV"""
        if self.records:
            df = pd.DataFrame(self.records, columns=self.columns)
            df.to_csv(self.save_path, index=False)
            print(f"\nDataset saved to {self.save_path}")
            print(f"Total records: {len(self.records)}")
        else:
            print("No records to save yet")

    def print_current_state(self):
        """Print real time system state for debugging"""
        state = self.get_system_state()
        print(f"\nSystem State:")
        print(f"Queue Length:         {state['queue_length']} workers")
        print(f"CPU Load:             {state['system_cpu_load']}%")
        print(f"Memory Available:     {state['system_memory_available']:.2f} GB")
        print(f"CPU Frequency:        {state['cpu_freq']} MHz")


# test the monitor
if __name__ == '__main__':
    monitor = SystemMonitor()
    
    print("Testing psutil monitor...")
    print("Reading Windows system state:\n")
    
    # test real state reading
    for i in range(5):
        monitor.print_current_state()
        time.sleep(1)
    
    # test real recording
    monitor.record_task(
        task_type='cpu',
        pid=os.getpid(),
        arrival_time=100,
        completion_time=250,
        response_time=150,
        scheduler_used='TEST',
        action_taken='NORMAL'
    )
    
    print("\nTest record saved successfully")
    monitor.save_dataset()