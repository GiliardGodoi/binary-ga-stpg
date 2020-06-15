"""
    class condition

    decorator to define a function like a stop condition


    O problema aqui é que o decorator define uma condição que poderá ser vista
    onde o módulo condition for instanciado.
    temos que ativamente ativar ou desativar um condição pela propriedade `active`
    Isso permite definir diversas condições para o algortimo.

    Mas a abordagem com gerenciado de contexto podemos definir uma condição somente
    para um processo com

    with TimeLimitConstrain(4000):
        pop.evol(evo, n=10_000)

    and Its done!

    It'd be nice if we could do something like:

    with (TimeLimitConstrain(4000),
        BestKnownReached(163)) :

        pop.evol(evo, n=10_000)


"""
class StopConditionReached(Exception):
    pass

class condition:

    conditions = set()

    def __init__(self, func, msg=None, active=True):
        self.func = func
        self.msg = msg or func.__name__
        self.active = active

    def __str__(self):
        return f"Condition({self.msg}, activate : {self.active})"

    def __repr__(self):
        return f"Condition({self.msg}, activate : {self.active})"

    def __call__(self, population):
        return self.func(population)

    def __enter__(self):
        if self.active:
            self.conditions.add(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            self.conditions.remove(self)

    @classmethod
    def check(cls, population):
        for condition in cls.conditions:
            if condition and not condition(population):
                raise StopConditionReached(condition.msg)

        return True

    @classmethod
    def params(cls, msg="stop condition reached", active=True):

        def decorator(func):
            return condition(func, msg=None, active=active)

        return decorator