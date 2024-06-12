import time

def time_check(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"          ============================== \n           [{func}] took: {elapsed:.6f} seconds\n          ============================== ")
        return result, elapsed
    return wrapper