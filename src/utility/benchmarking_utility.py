import time
from typing import Callable

def evaluate_time(
        func: Callable
):
    def compute_time(
            *args,
            **kwargs
    ):
        start_time = time.perf_counter()
        func_result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time} seconds')
        return func_result

    return compute_time