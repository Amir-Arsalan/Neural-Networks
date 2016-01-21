'''
Created on Dec 11, 2015

@author: amirarsa
'''
# from theano import function, config, shared, sandbox
# import theano.tensor as T
# import numpy
# import time
# 
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000
# 
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], T.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in xrange(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')

# from __future__ import print_function
# 
# import numpy as np
# import matplotlib.pyplot as plt
# #import theano
# # By convention, the tensor submodule is loaded as T
# import theano.tensor as T
# from theano import shared
# import theano.sandbox.cuda.basic_ops as sb
# from theano import function
# from theano.printing import min_informative_str

def data_loader():
    return np.load('/home/amir/workspace/theanoL/src/mnist.pkl')


# mnist = data_loader()
# trainX, trainY = [mnist[0][0], mnist[0][1]]
# validX, validY = [mnist[1][0], mnist[1][1]]
# testX, testY = [mnist[2][0], mnist[2][1]]
# 
# num_input = trainX.shape[1]
# num_h1 = 10
# 
# 
# class logistic(object):
#     def __init__(self):
#         self.w = shared(np.zeros((num_input, num_h1), dtype=T.config.floatX), name = 'w', borrow=True)
#         self.b = shared(np.zeros((10, 1), dtype=T.config.floatX), name = 'b', borrow = True, broadcastable=(False, True))
#     
#     def forward(self, x):
#         return self.softmax(T.dot(x, self.w) + self.b)
#     
#     def softmax(self, a):
#         return T.nnet.softmax(a)
# 
# logReg = logistic()
# logReg.forward(trainX[0:100])


import time
import theano
import theano.tensor as T
import numpy as np
from theano import shared, function, Out
import gzip, cPickle
import theano.sandbox.cuda.basic_ops as sbasic


def computeMemoryUsage(data, dataType):
    """
    Computes how much memory will get allocated for the data on the GPU.
    It is assumed that data has labels
    
    DataType's valid options are 16, 32 and 64 corresponding to float16. float32 and float64 respectively    
    """
    numDataPoints = data.shape[0]
    print "Current free memory on the GPU is %s GBs" % str(sbasic.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024)
    print "Free GPU memory after data transfer is ~%s GBs with float%d data type" % (str(sbasic.cuda_ndarray.cuda_ndarray.mem_info()[0]/1024./1024/1024 - numDataPoints * 784 * dataType / 8 / 1024./1024/1024 + numDataPoints * dataType / 8 / 1024./1024/1024), dataType)

def storeDataAsSharedVar(data_xy, borrow = True):
    """
    Stores the given data set on shared variables
    
    Inputs:
    data_xy: A tuple (x, y). x is an N x M numpy ndarray with N the data points and M the features.
    y is a numpy 1D vector of length N containining the number to which the corresponding data point belongs
    
    Outputs: 
    A tuple (x, y) of the data set stored as shared variables
    """
    data_x, data_y = data_xy
    shared_x = shared(np.asarray(np.r_[data_x], dtype = T.config.floatX), borrow = borrow)
    shared_y = shared(np.asarray(data_y, dtype = T.config.floatX), borrow = borrow)
    return shared_x, T.cast(shared_y, 'int16')

def loadMNIST(dir = '/home/amir/workspace/theanoL/src/mnist.pkl.gz'):
    """
    Loads MNIST as tuples of (x, y) as shared variables and returns them as a list of tuples 
    """
    f = gzip.open(dir)
    train_xy, valid_xy, test_xy = cPickle.load(f)
    f.close()
    trainSharedData = storeDataAsSharedVar(train_xy)
    validSharedData = storeDataAsSharedVar(valid_xy)
    testSharedData = storeDataAsSharedVar(test_xy)
    return [trainSharedData, validSharedData, testSharedData]

# 
# data = loadMNIST()
# train_x, train_y = storeDataAsSharedVar(train)
# valid_x, valid_y = storeDataAsSharedVar(valid)
# test_x, test_y = storeDataAsSharedVar(test)
# 
# N = np.asscalar(train_x.shape[0].eval())
# inputs = np.asscalar(train_x.shape[1].eval())
# outputs = 10
# 
# data = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]





class layer(object):
    def __init__(self, inputs, nInput, nOutput, lastLayer = False, activation=None, weightInitMode = 'normal'):
        
        if(activation is not None):
            self.activation = activation.strip().lower()
        else:
            self.activation = activation
        self.weightInitMode = weightInitMode.strip().lower()
        
        self.x = inputs
        self.nInput = nInput
        self.nOutput = nOutput
        self.lastLayer = lastLayer
        #self.w = shared(np.random.randn(nInput, nOutput).astype(T.config.floatX), borrow = True)
        #self.w = shared(np.random.uniform(-4 *np.sqrt(6. / (self.nInput + self.nOutput)), 4 * np.sqrt(6. / (self.nInput + self.nOutput)), (self.nInput, self.nOutput)).astype(T.config.floatX), borrow = True)
        #self.b = shared(np.zeros((self.nOutput), dtype = T.config.floatX), borrow = True)
        self.weightInit()
        self.forwardPass()
        self.params = [self.w, self.b]
        
    
    def weightInit(self):        
        if(self.weightInitMode == 'normal'):
            self.w = shared(np.random.normal(0., 0.01, (self.nInput, self.nOutput)).astype(T.config.floatX), borrow = True)
            if(self.activation == 'relu'):
                self.b = shared(np.ones((self.nOutput), dtype = T.config.floatX) * 0.1, borrow = True)
            elif(self.activation != 'relu' or self.lastLayer == False):
                self.b = shared(np.zeros((self.nOutput), dtype = T.config.floatX), borrow = True)
        
        elif(self.weightInitMode == 'xavier'):
#             if(self.lastLayer == False):
            if(self.activation == 'sigmoid'):
                self.w = shared(np.random.uniform(-4*np.sqrt(6. / (self.nInput + self.nOutput)), 4*np.sqrt(6. / (self.nInput + self.nOutput)), (self.nInput, self.nOutput)).astype(T.config.floatX), borrow = True)
            else:
                """
                For tanh or any other activation function
                """
                self.w = shared(np.random.uniform(-np.sqrt(6. / (self.nInput + self.nOutput)), np.sqrt(6. / (self.nInput + self.nOutput)), (self.nInput, self.nOutput)).astype(T.config.floatX), borrow = True)
            if(self.activation == 'relu'):
                self.b = shared(np.ones((self.nOutput), dtype = T.config.floatX) * 0.1, borrow = True)
            else:
                self.b = shared(np.zeros((self.nOutput), dtype = T.config.floatX), borrow = True)
    
    def forwardPass(self):
        self.forwardProp = T.dot(self.x, self.w) + self.b
        if (self.lastLayer == False):
            if(self.activation == "relu"):
                self.forwardProp = T.switch(self.forwardProp < 0, 0.01*self.forwardProp, self.forwardProp)
            elif(self.activation == 'sigmoid'):
                self.forwardProp = T.nnet.sigmoid(self.forwardProp)
        else:
            self.forwardProp = T.nnet.softmax(self.forwardProp)
            self.prediction = T.argmax(self.forwardProp, axis = 1)


class MLP(object):
    def __init__(self, inputs, nInput, nHidden, nOutput, dropoutRates, activation='relu', weightInitMode = 'normal'):
        """
        nHidden could be a list containing the number of hidden units in each hidden layer.
        It could also be empty. If empty, Logistic Regression classifier will be implemented.
        """
        self.x = inputs
        self.hLayers = []
        if(len(nHidden) > 0):
            self.hLayers.append(layer(self.x, nInput, nHidden[0], activation = activation, weightInitMode = weightInitMode)) #The first hidden layer
            if(len(nHidden) > 1):
                for i in range(1, len(nHidden)):
                    self.hLayers.append(layer(self.hLayers[i - 1].forwardProp, self.hLayers[i - 1].nOutput, nHidden[i], activation=activation, weightInitMode = weightInitMode))
            self.hLayers.append(layer(self.hLayers[len(self.hLayers) - 1].forwardProp, self.hLayers[len(self.hLayers) - 1].nOutput, nOutput, lastLayer = True, activation = activation)) #The output layer
        else:
            self.hLayers.append(layer(self.x, nInput, nOutput, lastLayer = True))      
        
        self.params = []
        for hL in self.hLayers:
            self.params += hL.params
        
        self.l2 = 0
        for hL in self.hLayers:
            self.l2 += T.sum(hL.w ** 2)
        
    def loss(self, y, regularization):
        return -T.mean(T.log(self.hLayers[len(self.hLayers) - 1].forwardProp)[T.arange(y.shape[0]), y]) + regularization * self.l2
    
    def error(self, y):
        return T.mean(T.neq(self.hLayers[len(self.hLayers) - 1].prediction, y))
        

def compileModel(data, nInputs, nOutputs, hiddenLayersSize = [1200, 1200], dropoutRates = [0.2, 0.5, 0.5],
                  activation = 'relu', weightInitMode = 'normal', regularizer = 0.0001):
    """
    Creates a symbolic model given the specified parameters using Theano
    
    Output:
    A list containing three the training, validation and test compiled functions of Theano
    """
    
    
    np.random.seed(815)
    
    x = T.matrix('x')
    y = T.wvector('y')
    learningRate = T.scalar('learningRate')
    regularization = T.scalar('regularization')
    
    #Data sets
    train_x, train_y = data[0]
    valid_x, valid_y = data[1]
    test_x, test_y = data[2]
    
    nnet = MLP(x, nInputs, hiddenLayersSize, nOutputs, dropoutRates = dropoutRates,
                activation = activation, weightInitMode = weightInitMode)
    
    loss = nnet.loss(y, regularization)
    error = nnet.error(y)
    
    gParams = T.grad(loss, nnet.params)
    
    weightUpdates = [(param, param - learningRate * gParam) for param, gParam in zip(nnet.params, gParams)]    
    
    
    batchIndicesVecctor = T.ivector('batchIndicesVecctor')
    trainF = function([batchIndicesVecctor, learningRate, regularization], Out(sbasic.gpu_from_host(loss), borrow = True), updates = weightUpdates, givens = {x: train_x[batchIndicesVecctor], y: train_y[batchIndicesVecctor]})
    validF = function([batchIndicesVecctor], Out(sbasic.gpu_from_host(T.cast(error, T.config.floatX)), borrow = True), givens = {x: valid_x[batchIndicesVecctor], y: valid_y[batchIndicesVecctor]})
    testF = function([batchIndicesVecctor], Out(sbasic.gpu_from_host(T.cast(error, T.config.floatX)), borrow = True), givens = {x: test_x[batchIndicesVecctor], y: test_y[batchIndicesVecctor]})
    
    return [trainF, validF, testF]
    
    

def mlp_sgd(Ns, modelFunctions, lr = 0.01, regularization = 0.0001, iters = 3000, batchSize = 20, randShuffle = False,
            drop_out = [0.2, 0.05]):
    
    """
    Does stochastic gradient descent on the input model.
    
    Inputs:
    Ns: A list containing the number of data points for each of the training, validation and test sets [N1, N2, N3]
    modelFunctions: A python list containing three complied Theano functions: train, validate, test
    
    Outputs:
    
    
    """
    
    np.random.seed(815)
    
    
    trainIndices = generateIndices(Ns[0])
    validIndices = generateIndices(Ns[1])
    testIndices = generateIndices(Ns[2])
    trainF = modelFunctions[0]
    validF = modelFunctions[1]
    testF = modelFunctions[2]
    
    t1 = time.time()
    for i in range(1, iters + 1):
        listOfBatchIndices = genPermutedBatchIndices(trainIndices, batchSize, randShuffle)
        tLoss = []
        for j in range(len(listOfBatchIndices)):
            tLoss.append(trainF(listOfBatchIndices[j], lr, regularization))
#         for j in range(len(trainIndices) - 1):
#             tLoss.append(np.asarray(trainF(trainIndices[j], trainIndices[j + 1], lr, 0)))
        print "The loss is %f on iteration %d" % (np.mean([np.asarray(nll) for nll in tLoss]), i - 1)
#         lr = max(0.008, lr * 0.985)
        #lr = learningRateUpdate(lr, mode='exponential')
#         print "Learning Rate has been Changed to %f" % lr
        if(i % 1 == 0):
            vLoss = []
            listOfBatchIndices = genPermutedBatchIndices(validIndices, batchSize, False)
            for j in xrange(len(listOfBatchIndices) - 1):
                vLoss.append(np.asarray(validF(listOfBatchIndices[j])))
#             for j in range(len(validIndices) - 1):
#                 vLoss.append(np.asarray(validF(validIndices[j], validIndices[j + 1])))
            print "The error is %f on iteration %d" % (np.mean([np.asarray(nll) for nll in vLoss]), i - 1)
#             if(i % 20 == 0):
#                 batchSize = int(min(round(batchSize * 1.3), train_x.shape[0].eval()/10))
#                 print "Batch size changed to %d" % batchSize
#                 lr = max(0.01, lr*0.88)
#                 print "Learning Rate has been Changed to %f" % lr
            print ''
    tLoss = []
    listOfBatchIndices = genPermutedBatchIndices(testIndices, batchSize, False)
    for j in range(len(listOfBatchIndices) - 1):
        tLoss.append(np.asarray(testF(listOfBatchIndices[j])))
    print "Test error is %f" % np.mean([np.asarray(nll) for nll in tLoss])
    print "Took %f seconds" % (time.time() - t1)
    print ''

def learningRateUpdate(learningRate,t ,step = 1, mode = 'exp', k = 10, ratio = 0.998):
    """
    Returns a new learning rate given the mode
    
    Inputs:
    learningRate = The previous learning rate
    t = Epoch number (time)
    step = Update the weights every 'step' numbers of iteration
    mode = Either of 'exp', 'step' or '1/t'
    k = hyper-paramter for exponential decay learning rate; when mode = 'exp'
    ratio = The new learning rate will be multiplied by step; if mode = 'stepwise'
    
    Output:
    New learning rate
    """
    
    return 

def generateIndices(numDataPoints):
    """
    Generates indices corresponding to each data point given a batchSize
    
    Inputs:
    numDataPoints: An integer number
    
    Output:
    indices: A numpy array of int32 containing integers 0 to numDataPoints
    """
    return np.asarray(np.append(np.arange(0, numDataPoints - 1), [numDataPoints - 1]), dtype='int32')

def genPermutedBatchIndices(arr, batchSize, randShuffle = True):
    """
    Generates a list of arrays given an array (arr) as input.
    The number of arrays inside the list is equivalent to total
    number of numbers in 'arr' divided by the number of batches
    
    Inputs:
    arr: numpy 1-D row array
    batchSize: size of examples in each list for stochastic gradient descent
    randShuffle: If True, the function generates permuted batch indices and vice versa otherwise
    
    Output:
    listOfBatchIndices: A list containing numpy arrays with length equivalent to number of batches
    The length of listOfNums = N / batchSize where N is arr.shape[0]
    """
    
    if(randShuffle):
        listOfBatchIndices = []
        np.random.shuffle(arr)
        slices = range(0, arr.shape[0] - 1, batchSize) + [arr.shape[0] - 1]
        for i in xrange(len(slices) - 1):
            listOfBatchIndices.append(arr[slices[i]:slices[i+1]])
    else:
        listOfBatchIndices = []
        slices = range(0, arr.shape[0] - 1, batchSize) + [arr.shape[0] - 1]
        for i in xrange(len(slices) - 1):
            listOfBatchIndices.append(arr[slices[i]:slices[i+1]])
    return listOfBatchIndices

if __name__ == "__main__":
    
    #Network parameters
    hiddenLayersSize = [500]
    dropoutRates = [0.2, 0.5, 0.5]
    
    #assert len(dropOutRates) - 1 == len(hiddenLayersSize)
    
    data = loadMNIST()
    nInputs = np.asscalar(data[0][0].shape[1].eval())
    nOutputs = np.unique(data[0][1].eval()).shape[0]
    Ns = [data[0][0].shape[0].eval(), data[1][0].shape[0].eval(), data[2][0].shape[0].eval()]
    modelFunctions = compileModel(data, nInputs, nOutputs, hiddenLayersSize = hiddenLayersSize, dropoutRates = dropoutRates,
                                   activation = 'relu', weightInitMode = 'normal')
    mlp_sgd(Ns, modelFunctions)
