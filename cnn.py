
import numpy as np
import imageStar
import layer


class CNN:

    def __init__(self,*args):
        nargs = len(args)
        if nargs == 4:
            self.Name = args[0]
            self.Layers = args[1]
            self.numLayers = len(args[1])
            self.InputSize = args[2]
            self.OutputSize = args[3]
        elif nargs == 0:
            self.Layers = []
            self.numLayers = 0
            self.InputSize = 0
            self.OutputSize = 0
        else:
            Exception("Invalid number of inputs, should be 0/4")

        self.reachMethod = "approx-star"    # reachable set computation scheme, default 'approx-star'
        self.relaxFactor = 0                # default - solve 100% LP optimization for finding bounds in 'approx-star' method
        self.reachOption = []               # parallel option, default - non-parallel computing
        self.numCores = 1                   # number of cores using in computation
        self.reachSet = {}                  # reachable set for each layers
        self.outputSet = []
        self.reachTime = []                 # computation time for each layers
        self.totalReachTime = 0             # total computation time
        self.features = {}                  # outputs of each layer in an evaluation
        self.dis_opt = []                   # display option = 'display' or []
        self.lp_solver = 'glpk'             # choose linprog as default LP solver for constructing reachable set

    def evaluate(self, x):
        '''
        Evaluation of this CNN
        :param x: input vector
        :return: output vector
        @features: output of all layers
        '''

        y = x
        for i in range(self.numLayers):
            y = self.Layers[i].evaluate(y)
            self.features.update({str(i): y})
        return y

    def reach(self, *args):

        nargs = len(args)
        if nargs == 1:
            inputSet = args[0]
        elif nargs == 2:
            inputSet = args[0]
            self.reachMethod = args[1]
            self.numCores = 1
        elif nargs == 3:
            inputSet = args[0]
            self.reachMethod = args[1]
            self.numCores = args[2]
        elif nargs == 4:
            inputSet = args[0]
            self.reachMethod = args[1]
            self.numCores = args[2]
            self.relaxFactor = args[3]
        elif nargs == 5:
            inputSet = args[0]
            self.reachMethod = args[1]
            self.numCores = args[2]
            self.relaxFactor = args[3]
            self.dis_opt = args[4]
        elif nargs == 6:
            inputSet = args[0]
            self.reachMethod = args[1]
            self.numCores = args[2]
            self.relaxFactor = args[3]
            self.dis_opt = args[4]
            self.lp_solver = args[5]
        else:
            raise Exception("Invalid number of input arguments, the number should be 1/2/3/4/5/6")

        rs = inputSet
        for i in range(1,self.numLayers+1):
            rs_new = self.Layers[i-1].reach(rs, self.reachMethod, self.reachOption, self.relaxFactor, self.dis_opt, self.lp_solver)
            rs = rs_new
            self.reachSet.update({str(i-1): rs_new})
        IS = rs_new
        self.outputSet = rs_new
        return IS

    def classify(self, *args):
        nargs = len(args)
        if nargs == 1:
            in_image = args[0]
        elif nargs == 2:
            in_image = args[0]
            method = args[1]
            numOfCores = 1
        elif nargs == 3:
            in_image = args[0]
            method = args[1]
            numOfCores = args[2]
        else:
            Exception("Invalid number of inputs, should be 1/2/3")

        if isinstance(in_image, imageStar.ImageStar):
            y = self.evaluate(in_image)
            y = y.reshape([self.OutputSize,1])
            label_id = max(y)[1]
        else:
            if nargs == 1:
                method = 'approx-star'
                numOfCores = 1

            self.reach(in_image, method, numOfCores)
            RS = self.outputSet
            n = len(RS)
            label_id = {}
            for i in range(n):
                rs = RS[i]
                new_rs = imageStar.ImageStar.reshape(rs, [self.outputSet[0],1,1])
                max_id = new_rs.get_localMax_index([1,1], [self.outputSet[0],1], 1)
                label_id.update({str(i): max_id[:,1]})

    def parse(self, *args):
        nargs = len(args)
        if nargs == 1:
            net = args[0]
            name = 'parsed_net'
        elif nargs == 2:
            net = args[0]
            name = args[1]
        else:
            Exception("Invalid number of input arguments, should be 1")

        n = len(net.Layers)
        LS = {}
        j = 0
        for i in range(n):
            L = net.Layers[i]
            if L == 'layer.ImageInputLayer':
                inputSize = L.InputSize
            elif L == 'layer.ClassificationOutputLayer':
                outputSize = L.OutputSize

            if L == 'layer.ImageInputLayer':
                Li = layer.ImageInputLayer.parse(L)
            elif L == 'layer.MergedConv2d':
                Li = layer.MergedConv2d.parse(L)
            elif L == 'layer.MaxPooling2d':
                Li = layer.MaxPooling2d.parse(L)
            elif L == 'layer.FC':
                Li = layer.FC.parse(L)

            j += 1
            LS.update({str(j): Li})

        net = CNN(name, LS, inputSize, outputSize)
        return net

    def evaluateRobustness(self, *args):
        nargs = len(args)
        if nargs == 3:
            in_images = args[0]
            correct_ids = args[1]
            method = args[2]
            numOfCores = 1
        elif nargs == 4:
            in_images = args[0]
            correct_ids = args[1]
            method = args[2]
            numOfCores =  args[3]

        N = len(in_images)
        if len(correct_ids) != N:
            Exception("Inconsistency between the number of correct_ids and the number of input sets")

        count = np.zeros((1,N))
        if method != 'exact-star':
            outputSets = self.reach(in_images, method, numOfCores)

            for i in range(N):
                count[i] = CNN.isRobust(outputSets[i], correct_ids[i])

        if method == 'exact-star':
            for i in range(N):
                outputSets = self.reach(in_images[i], method)
                M = len(outputSets)
                count1 = 0
                for j in range(M):
                    count1 += CNN.isRobust(outputSets[j], correct_ids[i])
                if count1 == M:
                    count[i] = 1

        r = sum(count)/N
        return r

    def isRobust(self, outputSet, correct_id):
        if correct_id > outputSet.numChannel | correct_id < 1:
            Exception("Invalid correct id")

        count = 0
        for i in range(outputSet.numChannel):
            if correct_id != i:
                if outputSet.is_p1_larger_p2([1, 1, i], [1, 1, correct_id]):
                    bool = 0
                    break
                else:
                    count += 1

        if count == outputSet.numChannel - 1:
            bool = 1

        return bool
