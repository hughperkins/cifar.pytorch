"""
Trains cifar on bn network

Usage:
  run.py [options]

Options:
  --save SAVE                  subdirectory to save logs [default: logs]
  --batchSize BATCHSIZE        batch size [default: 128]
  --learningRate LEARNINGRATE  learning rate [default: 1]
  --learningRateDecay LRDECAY  learning rate decay [default: 1e-7]
  --weightDecay WEIGHTDECAY    weightDecay [default: 0.0005]
  --momentum MOMENTUM          momentum [default: 0.9]
  --epoch_step EPOCHSTEP       epoch step [default: 25]
  --save_every SAVEEVERY       epochs between saves [default: 50]
  --model MODEL                model name [default: vgg_bn_drop]
  --max_epoch MAXEPOCH         maximum number of iterations [default: 300]
  --backend BACKEND            backend float|cuda|cl [default: cuda]
  --cudnnfastest CUDNNFASTEST  use cudnn 'fastest' mode y/n [default: y]
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
epoch_step = int(args['--epoch_step'])
max_epoch = int(args['--max_epoch'])
save = args['--save']
save_every = int(args['--save_every'])
learningRate = float(args['--learningRate'])
learningRateDecay = float(args['--learningRateDecay'])

# lua side params:
opt['weightDecay'] = float(args['--weightDecay'])
opt['momentum'] = float(args['--momentum'])
opt['model'] = args['--model']
opt['backend'] = args['--backend']
opt['cudnnfastest'] = args['--cudnnfastest'] == 'y'

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
Trainer = PyTorchHelpers.load_lua_class('train.lua', 'Trainer')
trainer = Trainer(opt)

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

def train(epoch, batchSize, learningRate):
  if epoch % 50 == 0:
    learningRate /= 2.0
    print('dropping learning rate to %s' % learningRate)
  epochLoss = 0
  batchesPerEpoch = NTrain // batchSize
  if devMode:
    batchesPerEpoch = 3  # impatient developer :-P
  last = time.time()
  for b in range(batchesPerEpoch):
    # draw samples
    indexes = np.random.randint(NTrain, size=(batchSize))

    batchInputs = trainData[indexes]
    batchLabels = trainLabels[indexes]

    loss = trainer.trainBatch(learningRate, batchInputs, batchLabels)
    now = time.time()
    duration = now - last
    last = now
    print('  epoch %s batch %s/%s loss %s time %s' %(epoch, b, batchesPerEpoch, loss, duration))
    epochLoss += loss

  return epochLoss

def test(epoch, batchSize):
  # evaluate on test data
  numTestBatches = NTest // batchSize
  if devMode:
    numTestBatches = 3  # impatient developer :-P
  testNumRight = 0
  testNumTotal = numTestBatches * batchSize
  for b in range(numTestBatches):
    batchInputs = testData[b * batchSize:(b+1) * batchSize]
    batchLabels = testLabels[b * batchSize:(b+1) * batchSize]
    pred = trainer.predict(batchInputs).asNumpyTensor().reshape(batchSize)
#    print('pred', pred)
#    print('batchLabels', batchLabels)
    batchCorrect = (pred == batchLabels).sum()
    testNumRight += batchCorrect

  testacc = testNumRight / testNumTotal * 100
  return testacc

for i in range(max_epoch):
  epoch = i + 1
  print('epoch', epoch)
  trainLoss = train(epoch, batchSize, learningRate)
  testacc = test(epoch, batchSize)
  print('epoch %s trainloss=%s testacc=%s' %
        (epoch, trainLoss, testacc))

  # save model every 50 epochs
  if epoch % save_every == 0:
    filename = join(save, 'model.net')
    print('==> saving model to %s' % filename)
    trainer.save(filename)

