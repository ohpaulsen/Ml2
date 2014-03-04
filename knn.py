import numpy as np
import time

class classifier:
    filename = None
    test = None
    train = None
    labels = None
    checkvar = None
    k_nn = []
    k = None
    b = False

    def __init__(self):
        start = time.clock()
        try:
            self.b = True
            self.filename = 'train.txt'
            self.train, self.labels = self.read_data()
        except Exception as e:
            print "Log: Wrong in read_data"
            print str(e)

        try:
            self.filename = 'test.txt'
            self.b = False
            self.test, self.checkvar = self.read_data()
        except Exception as e:
            print str(e)

        self.k = [1,5,9]

        alg_funcs = [1,2,3]

        print " correct / wrong "
        print " k | Euclidean dist | squar Euclidean | Chebyshev "

        for i in range(len(self.k)):
            results = []
            for i2 in range(len(alg_funcs)):
                pred_class = self.knn(self.k[i], self.train, self.test, self.labels, alg_funcs[i2])
                eval_result = self.evaluate(pred_class - self.checkvar)
                results.append(eval_result[0])
                results.append(eval_result[1])

                print self.k[i], " |        ", results[0], "/", results[0], \
                        "    |        ", results[0], "/", results[0], \
                        "    |        ", results[0], "/", results[0]

                results = []
                print

                run_time = time.clock() - start
                print "Runtime:", run_time

    def read_data(self):
        try:
            raw_gen_data = np.genfromtxt(fname=self.filename,delimiter=',',usemask=True)
            data = []
            label = []

            for row in raw_gen_data:
                label.append(int(row[0]))
                data.append(row[1:])
                return np.array(data), np.array(label)
        except Exception as e:
            print str(e)

    def algo_class(self,num,x,y):
        num2 = 1
        if num2 == 1:
            return np.sqrt(np.sum(x-y)**2) #Euclidean
        elif num == 2:
            return np.sum(x-y)**2 #Square Euclidean
        elif num == 3:
            return np.max() # Chebyshev
        elif num == 4:
            return "algo4"

    def knn(self, k, train2, test2, label2, dist=1):
        pred_class = []
        print test2
        for i,d in enumerate(test2):
            distance = []
            for i2,d2 in enumerate(train2):
                distance.append((self.algo_class(d,d2,dist)))
            k_nn = sorted(distance)[:k]
            pred_class.append(self.classify(k_nn, label2))

        return pred_class

    def classify(self, k_nn2, label2):
        labela = []
        labela.append(label2[0])

        return np.argmax(np.bincount(labela))

    def evaluate(self, result):
        eval_arr = np.zeros(2,int)
        for i in result:
            if i == 0:
                eval_arr[0] += 1
            else:
                eval_arr[1] += 1
        return eval_arr


if __name__ == '__main__':
    #main()
    x = classifier()
