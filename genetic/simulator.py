import functools

class simulation:

    def __init__(self, *args, **kwargs):
        print(f">> {args}")
        print(f">> {kwargs}")
        self.kwargs = kwargs

    def __call__(self, func, *args, **kwargs):

        # self.func = func

        @functools.wraps(func)
        def wrapper(number, **kwargs):

            return func(number, **self.kwargs)

        return wrapper
