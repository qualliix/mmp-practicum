from functools import wraps


def check_arguments(*t):
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            types = t
            a = args
            if len(types) < len(a):
                a = a[:len(types)]
            elif len(types) > len(a):
                raise TypeError
            for arg, type in zip(a, types):
                if not isinstance(arg, type):
                    raise TypeError
            return func(*args)
        return wrapper
    return decorator
