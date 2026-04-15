from locust import HttpUser, task, between
import random

class NginxUser(HttpUser):
    # wait between 0.1 and 0.5 seconds between requests
    # simulates real user behaviour
    wait_time = between(0.1, 0.5)

    @task(4)
    def cpu_task(self):
        """40% of requests are CPU bound"""
        start = self.environment.runner.stats.total.start_time
        with self.client.get(
            "/cpu-task",
            catch_response=True,
            name="CPU Bound Task"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(3)
    def memory_task(self):
        """30% of requests are memory bound"""
        with self.client.get(
            "/memory-task",
            catch_response=True,
            name="Memory Bound Task"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

    @task(3)
    def io_task(self):
        """30% of requests are I/O bound"""
        with self.client.get(
            "/io-task",
            catch_response=True,
            name="IO Bound Task"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")