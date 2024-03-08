import time





def time_complexity(func):
    def warp(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Time inference func {func.__name__}: {(time.time() - start):.3f} second')
        return result
    return warp

def add_func_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper