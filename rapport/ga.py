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
        """Doing the crossover and switching to new childerens """

        xory = self.random(0,2)
        randpos = self.random(0,len(self.chromosomes)-1)
        randpos2 = randpos
        while(randpos == randpos2):
            randpos2 = self.random(0,len(self.chromosomes)-1)

        if xory == 0:
            tempx = self.getvalue(randpos,xory)
            tempx2 = self.getvalue(randpos2,xory)
            self.addToChrom(self.getvalue(randpos,1),tempx)
            self.addToChrom(self.getvalue(randpos2,1),tempx2)
        else:
            tempx = self.getvalue(randpos,xory)
            tempx2 = self.getvalue(randpos2,xory)
            self.addToChrom(tempx,self.getvalue(0,randpos))
            self.addToChrom(tempx2,self.getvalue(0,randpos2))

    def getvalue(self,pos,xory):
        for i in range(len(self.chromosomes)):
            for i2 in range(0,1):
                if xory == i2:
                    if i == pos:
                        return self.chromosomes[i][i2]

    def addToChrom(self,x,y):
        self.chromosomes = np.append(self.chromosomes, (x,y))

    def mutations(self):
        """Mutiation procces where i spawn new values"""
        bol = False
        newdude = 0
        newdude2 = 0
        while bol == False:
            newdude = self.random(-10000,10000)
            newdude2 = self.random(-10000,10000)
            for i in range(len(self.chromosomes)):
                if(newdude == self.getvalue(i,0) or newdude2 == self.getvalue(i,1)):
                    pass;
                else:
                    bol = True
                    break
        self.addToChrom(newdude,newdude2)


    def fitness(self,x,y):
        return 75 * (y +x**2)**2 + (1 + x)**2


if __name__ == "__main__":
    x = GA()
    x.randomchroms()
    for i in range(0,1000):
        x.selection()
        x.crossover()
        x.mutations()

