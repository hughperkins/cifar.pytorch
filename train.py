"""
Trains cifar on residual network

Usage:
  run.py [options]

Options:
  --save SAVE                  subdirectory to save logs [default: logs]
  --batchSize BATCHSIZE        batch size [default: 128]
  --learningRate LEARNINGRATE learning rate [default: 1]
  --learningRateDecay LRDECAY learning rate decay [default: 1e-7]
  --weightDecay WEIGHTDECAY   weightDecay [default: 0.0005]
  --momentum                  momentum [default: 0.9]
  --epoch_step                epoch step [default: 25]
  --model                     model name [default: vgg_bn_drop]
  --max_epoch                 maximum number of iterations [default: 300]
  --backend                   backend [default: nn]
"""

from __future__ import print_function, division
import platform
import sys
import os
import random
import time
from os import path
from os.path import join
from docopt import docopt
import numpy as np
import PyTorchHelpers
pyversion = int(platform.python_version_tuple()[0])
if pyversion == 2:
  import cPickle
else:
  import pickle

args = docopt(__doc__)
opt = {}

# python-side params:
batchSize = int(args['--batchSize'])
epoch_step = int(args['--epoch_step']
max_epoch = int(args['--max_epoch'])
save = args['--save']
learningRate = float(args['--learningRate'])
learningRateDecay = float(args['--learningRateDecay'])

# lua side params:
opt['weightDecay'] = float(args['--weightDecay'])
opt['momentum'] = float(args['--momentum'])
opt['model'] = args['--model']
opt['backend'] = args['--backend']

data_dir = 'cifar-10-batches-py'
num_datafiles = 5
devMode = False
if 'DEVMODE' in os.environ and os.environ['DEVMODE'] == '1':
  devMode = True
  num_datafiles = 1 # cos I lack patience during dev :-P

inputPlanes = 3
inputWidth = 32
inputHeight = 32

def loadPickle(path):
  with open(path, 'rb') as f:
    if pyversion == 2:
      return cPickle.load(f)
    else:
      return {k.decode('utf-8'): v for k,v in pickle.load(f, encoding='bytes').items()}

def loadData(data_dir, num_datafiles):
  # load training data
  trainData = None
  trainLabels = None
  NTrain = None
  for i in range(num_datafiles):
    d = loadPickle(join(data_dir, 'data_batch_%s' % (i+1)))
    dataLength = d['data'].shape[0]
    NTrain = num_datafiles * dataLength
    if trainData is None:
      trainData = np.zeros((NTrain, inputPlanes, inputWidth, inputHeight), np.float32)
      trainLabels = np.zeros(NTrain, np.uint8)
    data = d['data'].reshape(dataLength, inputPlanes, inputWidth, inputHeight)
    trainData[i * dataLength:(i+1) * dataLength] = data
    trainLabels[i * dataLength:(i+1) * dataLength] = d['labels']

  # load test data
  d = loadPickle(join(data_dir, 'test_batch'))
  NTest = d['data'].shape[0]
  testData = np.zeros((NTest, inputPlanes, inputWidth, inputHeight), np.float32)
  testLabels = np.zeros(NTest, np.uint8)
  data = d['data'].reshape(dataLength, inputPlanes, inputWidth, inputHeight)
  testData[:] = data
  testLabels[:] = d['labels']

  return NTrain, trainData, trainLabels, NTest, testData, testLabels


# load the lua class
Train = PyTorchHelpers.load_lua_class('train.lua', 'Train')
train = ResidualTrainer(opt)
print('train', train)

# load data

NTrain, trainData, trainLabels, NTest, testData, testLabels = loadData(data_dir, num_datafiles)

print('data loaded :-)')

# I think the mean and std are over all data, altogether, not specific to planes or pixel location?
mean = trainData.mean()
std = trainData.std()

trainData -= mean
trainData /= std

testData -= mean
testData /= std

print('data normalized check new mean/std:')
print('  trainmean=%s trainstd=%s testmean=%s teststd=%s' %
      (trainData.mean(), trainData.std(), testData.mean(), testData.std()))

batchesPerEpoch = NTrain // batchSize
if devMode:
  batchesPerEpoch = 3  # impatient developer :-P
epoch = 0
while True:
#  print('epoch', epoch)
  learningRate = epochToLearningRate(epoch)
  epochLoss = 0
#  batchInputs 
  last = time.time()
  for b in range(batchesPerEpoch):
    # draw samples
    indexes = np.random.randint(NTrain, size=(batchSize))

    batchInputs = trainData[indexes]
    batchLabels = trainLabels[indexes]

    loss = train.trainBatch(learningRate, batchInputs, batchLabels)
    print('  epoch %s batch %s/%s loss %s' %(epoch, b, batchesPerEpoch, loss))
    epochLoss += loss

    if devMode:
      now = time.time()
      duration = now - last
      print('batch time', duration)
      last = now

  # evaluate on test data
  numTestBatches = NTest // batchSize
  if devMode:
    numTestBatches = 3  # impatient developer :-P
  testNumTop1Right = 0
  testNumTop5Right = 0
  testNumTotal = numTestBatches * batchSize
  for b in range(numTestBatches):
    batchInputs = testData[b * batchSize:(b+1) * batchSize]
    batchLabels = testLabels[b * batchSize:(b+1) * batchSize]
    res = train.predict(batchInputs)
    top1 = res['top1'].asNumpyTensor()
    top5 = res['top5'].asNumpyTensor()
    labelsTiled5 = np.tile(batchLabels.reshape(batchSize, 1), (1, 5))
    top1_correct = (top1 == batchLabels).sum()
    top5_correct = (top5 == labelsTiled5).sum()
    testNumTop1Right += top1_correct
    testNumTop5Right += top5_correct
#    print('correct top1=%s top5=%s', top1_correct, top5_correct)

  testtop1acc = testNumTop1Right / testNumTotal * 100
  testtop5acc = testNumTop5Right / testNumTotal * 100
  print('epoch %s trainloss=%s top1acc=%s top5acc=%s' %
        (epoch, epochLoss, testtop1acc, testtop5acc))
  epoch += 1

# from the lua:
#for i=1,opt.max_epoch do
#  train()
#  test()
#end

