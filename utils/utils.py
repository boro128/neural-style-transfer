from time import perf_counter
from datetime import timedelta


def time_execution(func):
    def wrapper():
        start_time = perf_counter()

        func()

        finish_time = perf_counter()
        difference = timedelta(seconds=finish_time-start_time)
        print(f"Execution time: {difference}")
    return wrapper
