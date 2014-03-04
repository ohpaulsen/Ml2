import numpy as np

class GA:
    use = 1 #or 2
    gen = 15
    chromosomeAnt = [4,10]
    percentCross = [30,50]
    mutations = [1,5]
    total = [100,1000]
    isize = chromosomeAnt[use]
    chromosomes = np.empty((gen,2)) # self.chromosomeAnt[self.use] ))

    def randomchroms(self):
        for i in range(0,self.gen):
            for i2 in range(0,2):
                self.chromosomes[i][i2] = self.random(1,10000)

    def random(self,start,stop):
        """generate random ints"""
        return int(np.random.randint(start,stop))

    def percent(self,percent,whole):
        return (percent * whole) / 100.0

    def selection(self):
        """getting the 4 best parrents"""
        results = np.zeros(shape=(0))
        best = []
        for i in range(0,len(self.chromosomes)):
            results = np.append(results,self.fitness(self.chromosomes[i][0],self.chromosomes[i][1]))
        best = results.argsort()[-4:][::-1]
        temp = np.empty((self.gen,2))
        for it in range(0,len(best)):
            for it2 in range(0,2):
                temp[it][it2] = self.chromosomes[best[it]][it2]
        self.chromosomes = temp

    def crossover(self):
        """lol"""

    def addToChrom(self,x,y):

        self.chromosomes = np.append(self.chromosomes, x,y)


    def generation(self):
        """lol"""

    def mutations(self):
        """lol"""

    def fitness(self,x,y):
        return 75 * (y +x**2)**2 + (1 + x)**2


if __name__ == "__main__":
    x = GA()
    x.randomchroms()
    x.selection()
    x.crossover()

