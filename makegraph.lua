require 'gnuplot'
require 'sys'
require 'os'
require 'paths'

local cmd = torch.CmdLine()
cmd:option('-savedir', 'logs', 'save dir')
local opt = cmd:parse(arg)

local savedir = opt.savedir
--print('savedir', savedir)

local trainValues = {}
local testValues = {}
local minEpoch = nil
local maxEpoch = nil
for line in io.lines(savedir .. '/res.log') do
--  print('line', line)
  local f = line:gmatch('([^=]+)')
  local linetype = f()
  if linetype == 'epoch' then
--    print('linetype', linetype)
    local epoch = tonumber(line:gmatch('epoch=(%d+)')())
    local trainacc = tonumber(line:gmatch('trainacc=([0-9.]+)')())
    local testacc = tonumber(line:gmatch('testacc=([0-9.]+)')())
--    print('epoch', epoch, 'trainacc', trainacc, 'testacc', testacc)
    trainValues[epoch] = trainacc
    testValues[epoch] = testacc
    minEpoch = minEpoch or epoch
    maxEpoch = epoch
  end
end

local numEpochs = maxEpoch - minEpoch + 1
local epochs = torch.FloatTensor(numEpochs)
local trainAcc = torch.FloatTensor(numEpochs)
local testAcc = torch.FloatTensor(numEpochs)
for epoch=minEpoch,maxEpoch do
  epochs[epoch - minEpoch + 1] = epoch
  trainAcc[epoch - minEpoch + 1] = trainValues[epoch]
  testAcc[epoch - minEpoch + 1] = testValues[epoch]
end
--print('trainAcc', trainAcc)

local myplot = gnuplot.pngfigure(savedir .. '/res.png')
gnuplot.title('CIFAR10 Accuracy')
local plots = {}
table.insert(plots, {"train", epochs, trainAcc, 'lines'})
table.insert(plots, {"test", epochs, testAcc, 'lines'})
gnuplot.plot(plots)
gnuplot.plotflush()
gnuplot.close(myplot)

