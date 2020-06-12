
class condition:

    conditions = set()

    def __init__(self, func):
        self.conditions.add(func)
        self.func = func

    def __call__(self,*args, **kwargs):
        return self.func(*args, **kwargs)

    @classmethod
    def check(cls, population):
        return all(condition(population) for condition in cls.conditions)