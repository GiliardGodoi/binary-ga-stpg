class BaseChromosome(object):
    '''Class provides a basic chromosome representation'''

    def __init__(self, genes):
        self.__genes = genes
        self.__cost = 0
        self.__fitness = 0
        self.normalized = False

    @property
    def genes(self):
        return self.__genes

    @genes.setter
    def genes(self, value):
        self.__genes = value

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, value):
        self.__cost = value
        self.__fitness = value
        self.normalized = False

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
        self.normalized = True

    def __len__(self):
        return 1

    def __str__(self):
        return str(self.genes)

    def __repr__(self):
        return self.__class__.__name__


class TreeBasedChromosome(BaseChromosome):

    def __init__(self, genes):
        super().__init__(genes)

    @property
    def graph(self):
        return self.genes

    @graph.setter
    def graph(self, newgraph):
        self.genes = newgraph