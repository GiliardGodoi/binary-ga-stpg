
class condition:

    conditions = set()

    def __init__(self, func, msg="Stop condition reached", active=True):
        if active:
            self.conditions.add(func)
        self.func = func
        self.msg = msg
        self._active = active

    def __call__(self,*args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        if value :
            self._active = True
            self.conditions.add(self.func)
        else :
            self._active = False
            self.conditions.remove(self.func)

    @classmethod
    def params(cls, msg="Stop condition reached", active=True):

        def decorator(func):
            print(func.__name__)
            return cls(func, msg=msg, active=active)

        return decorator


    @classmethod
    def check(cls, population):
        return all(condition(population) for condition in cls.conditions)