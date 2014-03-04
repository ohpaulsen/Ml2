import numpy as np
import scipy.spatial.distance as ssd
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
            print "Log: Wrong in read_data shit"
            print str(e)

        try:
            self.filename = 'test.txt'
            self.b = False
            self.test, self.checkvar = self.read_data()
        except Exception as e:
            print str(e)

        self.k = [1,5,9]

        alg_funcs = [1,2,3]

        print
        print " correct / wrong "
        print " k | Euclidean dist | squar Euclidean | Chebyshev "

        for i in range(len(self.k)):
            results = []
            for i2 in range(len(alg_funcs)):
                pred_class = self.knn(self.k[i], self.train, self.test, self.labels, alg_funcs[i2])
                eval_result = self.evaluate(pred_class - checkvar)
                results.append(eval_result[0])
                results.append(eval_result[1])

                print K[i], " |        ", results[0], "/", results[1], \
                        "    |        ", results[2], "/", results[3], \
                        "    |        ", results[4], "/", results[5]

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
            self.k_nn = []
            self.k_nn = sorted(distance)[:k]
            pred_class.append(self.classify(self.k_nn, label2))

        return pred_class

    def classify(self, k_nn2, label2):
        labela = []
        for dist, idx in k_nn2:
            labela.append(label2[idx])

        return np.argmax(np.bincount[labela])

    def ecaluate(self, result):
        eval_arr = np.zeros(2,int)
        for i in result:
            if i == 0:
                eval_arr[0] += 1
            else:
                eval_arr[1] += 1
        return eval_arr


def read_data(fn):
    """ read dataset and separate into input data
        and label data
    """

    raw_gen_data = np.genfromtxt(fname=fn,delimiter=',',usemask=True)

    data = []
    label = []

    for row in raw_gen_data:
        label.append(int(row[0]))
        data.append(row[1:])

    return np.array(data), np.array(label)




def knn(k, dtrain, dtest, dtr_label, dist=1):
    """ k-nearest neighbors """

    # initialize list to store predicted class
    pred_class = []
    # for each instance in data testing,
    # calculate distance in respect to data training
    for ii, di in enumerate(dtest):
        distances = []    # initialize list to store distance
        for ij, dj in enumerate(dtrain):
            # calculate distances
            distances.append((calc_dist(di,dj,dist), ij))
        # k-neighbors
        k_nn = sorted(distances)[:k]
        # predict the class for the instance
        pred_class.append(classify(k_nn, dtr_label))

    # return prediction class
    return pred_class
def calc_dist(di,dj,i=1):
    """ Distance calculation for every
        distance functions in use"""
    if i == 1:
        return ssd.euclidean(di,dj) # built-in Euclidean fn
    elif i == 2:
        return ssd.cityblock(di,dj) # built-in Manhattan fn
    elif i == 3:
        return ssd.cosine(di,dj)    # built-in Cosine fn

def classify(k_nn, dtr_label):
    """ Classify instance data test into class"""

    dlabel = []
    for dist, idx in k_nn:
        # retrieve label class and store into dlabel
        dlabel.append(dtr_label[idx])

    # return prediction class
    return np.argmax(np.bincount(dlabel))

def evaluate(result):
    """ Evaluate the prediction class"""

    # create eval result array to store evaluation result
    eval_result = np.zeros(2,int)
    for x in result:
        # increment the correct prediction by 1
        if x == 0:
            eval_result[0] += 1
        # increment the wrong prediction by 1
        else:
            eval_result[1] += 1
    # return evaluation result
    return eval_result

def main():
    """ k-nearest neighbors classifier """

    # initialize runtime
    start = time.clock()

    # data tests, 1 = breast cancer data test,
    # 2 = iris data test
    data_tests = [1]

    for d in data_tests:
        if d == 1:
            # read dataset of breast cancer
            dtrain, dtr_label = read_data('train.txt')
            dtest, true_class = read_data('test.txt')
        else:
            # read dataset of breast cancer
            dtrain, dtr_label = read_data('iris-train.csv')
            dtest, true_class = read_data('iris-test.csv')

        # initialize K
        K = [1,3,7,11]

        # distance function for euclidean (1), manhattan (2),
        # and cosine (3)
        dist_fn = [1,2,3]

        if d == 1:
            print "k-NN classification results for breast cancer data set:"
        else:
            print "k-NN classification results for iris data set:"

        print
        print "    Number of correct / wrong classified test records"
        print "k  | Euclidean dist | Manhattan dist | Cosine dist"

        # run knn classifier for each k and distance function
        for i in range(len(K)):
            # classification result for each distance function
            results = []
            for j in range(len(dist_fn)):
                # predict the data test into class
                pred_class = knn(K[i], dtrain, dtest, dtr_label, dist_fn[j])
                # evaluate the predicted result
                eval_result = evaluate(pred_class-true_class)
                # assign the evaluated result into classification result
                results.append(eval_result[0])
                results.append(eval_result[1])

            # print the classification result into the screen
            print K[i], " |     ", results[0], "/", results[1], \
                "    |    ", results[2], "/", results[3], \
                "     |    ", results[4], "/", results[5]
            results = []
        print

    # retrieve
    run_time = time.clock() - start
    print "Runtime:", run_time

if __name__ == '__main__':
    #main()
    classifier()
