from functools import wraps


def substitutive(func):
    @wraps(func)
    def wrapper(*args):
        accumulated_args = list(args)
        if len(accumulated_args) < func.__code__.co_argcount:
            @wraps(func)
            def inner(*remaining_args):
                nonlocal accumulated_args
                accumulated_args.extend(remaining_args)
                if len(accumulated_args) < func.__code__.co_argcount:
                    return inner
                else:
                    result = func(*accumulated_args)
                    accumulated_args = accumulated_args[:len(args)]
                    return result
            return inner
        else:
            return func(*accumulated_args)
    return wrapper
