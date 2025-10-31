import time
from functools import wraps

def time_it(iterations=100, label="Time"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # warm-up
            result = func(*args, **kwargs)  
            start = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            end = time.time()
            print(f"{label}: {end - start:.4f} sec for {iterations} iterations")
            return result
        return wrapper
    return decorator